# src/langchain_rpa_simkit/__main__.py
import sys
from langchain_rpa_simkit.simple_chain import plan

def main():
    task = " ".join(sys.argv[1:]) or "demo task"
    print(plan(task))

if __name__ == "__main__":
    main()

