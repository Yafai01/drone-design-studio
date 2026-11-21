"""
Interactive command-line front end for drone_design.py.

Run:  python run_designer.py
Press Enter to accept the default shown in [brackets]. Type q at any prompt to quit.
"""
import sys

print("Loading drone design model...")
from drone_design import design_drone, check_build  # noqa: E402


class Quit(Exception):
    pass


def ask(prompt, default=None):
    """Read one line; Enter returns the default, 'q' quits."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"{prompt}{suffix}: ").replace("﻿", "").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise Quit
    if raw.lower() in ("q", "quit", "exit"):
        raise Quit
    return raw if raw else (str(default) if default is not None else "")


def ask_choice(prompt, options, default):
    """options: dict mapping accepted inputs -> canonical value."""
    while True:
        raw = ask(prompt, default).lower()
        if raw in options:
            return options[raw]
        print(f"  Please enter one of: {', '.join(sorted(set(options)))}")


def ask_number(prompt, default, lo, hi, integer=False, allow_none=False):
    while True:
        raw = ask(prompt, default)
        if allow_none and raw.lower() in ("none", "no", "-", "0"):
            return None
        try:
            val = int(raw) if integer else float(raw)
        except ValueError:
            print(f"  Please enter a {'whole number' if integer else 'number'}.")
            continue
        if not lo <= val <= hi:
            print(f"  Please enter a value between {lo:g} and {hi:g}.")
            continue
        return val


def run_design(kind):
    print("\n--- Design a new", "multirotor" if kind == "multirotor" else "fixed-wing", "---")
    if kind == "multirotor":
        payload = ask_number("Payload (kg)", 0.5, 0, 25)
        flight = ask_number("Minimum flight time (min)", 20, 1, 180)
        size = ask_number("Max wheelbase (mm, or 'none')", 650, 100, 3000, allow_none=True)
    else:
        payload = ask_number("Payload (kg)", 1.0, 0, 10)
        flight = ask_number("Minimum flight time (min)", 60, 1, 600)
        size = ask_number("Max wingspan (m, or 'none')", 2.5, 0.6, 4.5, allow_none=True)
    goal = ask_choice("Goal: flight_time (longest flight) or light (lightest build)",
                      {"flight_time": "flight_time", "f": "flight_time", "flight": "flight_time",
                       "light": "light", "l": "light"},
                      "flight_time" if kind == "multirotor" else "light")
    print("\nSearching candidate designs (this takes a few seconds)...")
    design_drone(kind, payload_kg=payload, min_flight_min=flight, max_size=size, goal=goal)


def run_check(kind):
    print("\n--- Check an existing", "multirotor" if kind == "multirotor" else "fixed-wing", "build ---")
    parts = {}
    if kind == "multirotor":
        parts["n_motors"] = ask_choice("Number of motors (4, 6 or 8)", {"4": 4, "6": 6, "8": 8}, 4)
        d_def, p_def, kv_def, s_def, mah_def, pay_def = 10, 4.5, 920, 3, 5000, 0
    else:
        parts["span_m"] = ask_number("Wingspan (m)", 2.0, 0.3, 6)
        parts["aspect"] = ask_number("Aspect ratio", 10, 3, 20)
        d_def, p_def, kv_def, s_def, mah_def, pay_def = 12, 6, 800, 4, 5000, 0.5
    parts["D_in"] = ask_number("Prop diameter (in)", d_def, 2, 30)
    parts["P_in"] = ask_number("Prop pitch (in)", p_def, 1, 30)
    parts["kv"] = ask_number("Motor KV", kv_def, 50, 5000, integer=True)
    parts["cells"] = ask_number("Battery cells (S)", s_def, 1, 14, integer=True)
    parts["mah"] = ask_number("Battery capacity (mAh)", mah_def, 100, 50000, integer=True)
    parts["payload_kg"] = ask_number("Payload (kg)", pay_def, 0, 25)
    print()
    check_build(kind, **parts)


def main():
    print("\n=== Drone Design Assistant ===  (Enter = default, q = quit)")
    while True:
        try:
            kind = ask_choice("\nDrone type: multirotor or fixedwing",
                              {"multirotor": "multirotor", "m": "multirotor", "multi": "multirotor",
                               "fixedwing": "fixedwing", "f": "fixedwing", "fixed": "fixedwing",
                               "fixed-wing": "fixedwing"}, "multirotor")
            mode = ask_choice("Mode: design (new drone) or check (existing build)",
                              {"design": "design", "d": "design", "check": "check", "c": "check"},
                              "design")
            (run_design if mode == "design" else run_check)(kind)
            again = ask_choice("\nRun another? (y/n)", {"y": True, "yes": True, "n": False, "no": False}, "y")
            if not again:
                break
        except Quit:
            break
    print("Bye.")


if __name__ == "__main__":
    sys.exit(main())
