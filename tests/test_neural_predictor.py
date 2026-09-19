"""Behaviour checks for complete-word scoring and separate answer options."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_predictor import NeuralWordRanker

model = NeuralWordRanker("0.6B")
model.check_scoring()
phrase = "we went to the restaurant for"
free = model.predict(phrase, ["dinner", "uncharacteristically"])
with_choices = model.predict(phrase, ["dinner", "uncharacteristically"],
    ["Lunch", "dinner", "coffee", "tea"])
assert free["shortlist"] == with_choices["shortlist"]
assert free["words"] == with_choices["words"]
assert len(free["words"]) == 3
assert set(with_choices["choices"]) == {"lunch", "dinner", "coffee", "tea"}
try:
    model.predict(phrase, choices=["two words"])
except ValueError:
    pass
else:
    raise AssertionError("Multiple-word choices should be rejected.")
print("PASS: supplied choices do not alter the free shortlist or its top three; input validation works.")
