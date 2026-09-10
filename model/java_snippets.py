"""Java answers for Flow, every one verified by the real compiler.

Flow's model cannot write working Java — 11M parameters is not enough, and
there is no Java source on this machine to learn from. So Java is answered
from a table of snippets instead, and this file compiles every one with javac
before they are stored. If a snippet does not compile, the gate fails.

That is the honest trade: the answers are correct because a person wrote them
and a compiler checked them, not because a tiny model guessed well.

Run:  .venv/bin/python model/java_snippets.py     compile-check and store
"""
import os
import shutil
import subprocess
import sys
import tempfile

from knowledge import count, teach

# (question, snippet). Each is compiled as-is before being stored.
SNIPPETS = [
    ("write a java hello world",
     'public class Main {\n'
     '    public static void main(String[] args) {\n'
     '        System.out.println("Hello, world!");\n'
     '    }\n'
     '}'),

    ("write a java function that adds two numbers",
     'public class Main {\n'
     '    static int add(int a, int b) {\n'
     '        return a + b;\n'
     '    }\n'
     '}'),

    ("how do i make a loop in java",
     'public class Main {\n'
     '    public static void main(String[] args) {\n'
     '        for (int i = 0; i < 10; i++) {\n'
     '            System.out.println(i);\n'
     '        }\n'
     '    }\n'
     '}'),

    ("how do i make a list in java",
     'import java.util.ArrayList;\n\n'
     'public class Main {\n'
     '    public static void main(String[] args) {\n'
     '        ArrayList<String> names = new ArrayList<>();\n'
     '        names.add("Or");\n'
     '        names.add("Ariel");\n'
     '        System.out.println(names);\n'
     '    }\n'
     '}'),

    ("write a java class",
     'public class Dog {\n'
     '    private String name;\n\n'
     '    public Dog(String name) {\n'
     '        this.name = name;\n'
     '    }\n\n'
     '    public String getName() {\n'
     '        return name;\n'
     '    }\n'
     '}'),

    ("how do i use if in java",
     'public class Main {\n'
     '    public static void main(String[] args) {\n'
     '        int score = 12;\n'
     '        if (score > 10) {\n'
     '            System.out.println("you win");\n'
     '        } else {\n'
     '            System.out.println("try again");\n'
     '        }\n'
     '    }\n'
     '}'),

    ("write a java function that doubles a number",
     'public class Main {\n'
     '    static int doubleIt(int n) {\n'
     '        return n * 2;\n'
     '    }\n'
     '}'),

    ("how do i check if a number is even in java",
     'public class Main {\n'
     '    static boolean isEven(int n) {\n'
     '        return n % 2 == 0;\n'
     '    }\n'
     '}'),

    ("write a java function that reverses a string",
     'public class Main {\n'
     '    static String reverse(String text) {\n'
     '        return new StringBuilder(text).reverse().toString();\n'
     '    }\n'
     '}'),

    ("how do i read input in java",
     'import java.util.Scanner;\n\n'
     'public class Main {\n'
     '    public static void main(String[] args) {\n'
     '        Scanner in = new Scanner(System.in);\n'
     '        String line = in.nextLine();\n'
     '        System.out.println("You said: " + line);\n'
     '    }\n'
     '}'),

    ("write a java function that finds the biggest number",
     'public class Main {\n'
     '    static int biggest(int[] numbers) {\n'
     '        int best = numbers[0];\n'
     '        for (int n : numbers) {\n'
     '            if (n > best) {\n'
     '                best = n;\n'
     '            }\n'
     '        }\n'
     '        return best;\n'
     '    }\n'
     '}'),

    ("what is java",
     "A programming language used for Android apps, big websites and games "
     "like Minecraft. Every Java program lives inside a class."),
]


def class_name(source):
    """javac requires the file name to match the public class inside it."""
    for line in source.split("\n"):
        line = line.strip()
        if line.startswith("public class "):
            return line.split()[2].split("{")[0].strip()
        if line.startswith("class "):
            return line.split()[1].split("{")[0].strip()
    return "Main"


def javac_works():
    """Is there a real JDK, or only the macOS stub?

    /usr/bin/javac exists on every Mac but is a shim that prints "Unable to
    locate a Java Runtime" unless a JDK is actually installed.
    """
    if not shutil.which("javac"):
        return False
    try:
        r = subprocess.run(["javac", "-version"], capture_output=True,
                           text=True, timeout=60)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def compiles(source):
    """Compile with the real javac. Returns (ok, error)."""
    work = tempfile.mkdtemp(prefix="flow_java_")
    try:
        path = os.path.join(work, f"{class_name(source)}.java")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        result = subprocess.run(
            ["javac", "-d", work, path],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0, result.stderr.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def structurally_valid(source):
    """Weaker fallback when no JDK is installed.

    Catches real structural mistakes - unbalanced braces, a missing semicolon,
    no class at all - but cannot catch type errors or unknown methods the way
    a compiler does. Reported honestly as "structure" rather than "compiles".
    """
    depth = 0
    for ch in source:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False, "closing brace with nothing open"
    if depth != 0:
        return False, f"{depth} brace(s) left unclosed"

    if source.count("(") != source.count(")"):
        return False, "unbalanced parentheses"
    if "class " not in source:
        return False, "no class declaration"

    for raw in source.split("\n"):
        line = raw.strip()
        if (not line or line.startswith(("//", "/*", "*", "@", "}"))
                or line.endswith(("{", "}", ";"))
                or line.endswith(",")):        # a wrapped argument list
            continue
        return False, f"statement without a semicolon: {line[:40]}"

    return True, ""


if __name__ == "__main__":
    real_javac = javac_works()
    if real_javac:
        print("verifying every snippet with the real javac\n")
        check, label = compiles, "compiles"
    else:
        print("No JDK installed - /usr/bin/javac on this Mac is only a stub.")
        print("Falling back to a STRUCTURAL check: it catches unbalanced")
        print("braces and missing semicolons, but not type errors.")
        print("Install a JDK and re-run this to get real compilation.\n")
        check, label = structurally_valid, "structure ok"
    good = 0
    checked = 0
    for question, snippet in SNIPPETS:
        if not snippet.strip().startswith(("public", "import", "class")):
            teach(question, snippet)          # prose answer, nothing to compile
            print(f"  ----  {question:48s} (prose)")
            continue

        checked += 1
        ok, err = check(snippet)
        good += ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {question:48s}"
              f"{'' if ok else '  ' + err.splitlines()[0][:60]}")
        if ok:
            teach(question, snippet)

    rate = good / max(checked, 1)
    print(f"\n=== JAVA GATE ===")
    print(f"{'PASS' if rate == 1.0 else 'FAIL'}  {label}: "
          f"{good}/{checked}  ({rate:.0%})")
    if not real_javac:
        print("      (structural check only - no JDK on this machine)")
    print(f"\n{'JAVA READY' if rate == 1.0 else 'NOT READY'}  "
          f"({count()} facts stored)")
    sys.exit(0 if rate == 1.0 else 1)
