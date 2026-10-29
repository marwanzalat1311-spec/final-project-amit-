"""Try a trained model from the command line.

    python predict.py "chest pain spreading to my left arm and sweating" --age 60 --gender male
    python predict.py "sore throat, fever and swollen glands" --age 20
    python predict.py --interactive
    python predict.py "..." --model tfidf_logreg_v1
"""
import argparse

from modeling.predictor import Predictor


def show(res):
    print()
    for r in res["predictions"]:
        flag = "  [EMERGENCY]" if r["emergency"] else ""
        print(f"  {r['rank']}. {r['condition']:<45} {r['probability']*100:5.1f}%{flag}")
    top = res["predictions"][0]
    if top["warnings"]:
        print(f"\n  Warnings for {top['condition']}:")
        for w in top["warnings"][:3]:
            print("   -", w)
    if top["recommendations"]:
        print(f"  Recommendations:")
        for w in top["recommendations"][:3]:
            print("   -", w)
    if res["emergency_alert"]:
        print("\n  !!", res["emergency_message"])
    print("\n ", res["disclaimer"], "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?")
    p.add_argument("--age", type=int)
    p.add_argument("--gender", choices=["female", "male"])
    p.add_argument("--model", default="tfidf_logreg_v1")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--interactive", action="store_true")
    a = p.parse_args()

    pred = Predictor(model_name=a.model)
    if a.interactive or not a.text:
        print("Describe symptoms (empty line to quit).")
        while True:
            text = input("\nSymptoms: ").strip()
            if not text:
                break
            age = input("Age (optional): ").strip()
            gender = input("Gender female/male (optional): ").strip().lower() or None
            show(pred.predict(text, int(age) if age.isdigit() else None,
                              gender if gender in ("female", "male") else None, a.top))
    else:
        show(pred.predict(a.text, a.age, a.gender, a.top))


if __name__ == "__main__":
    main()
