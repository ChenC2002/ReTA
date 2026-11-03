# Validation

Run the offline software checks with:

```bash
python -m unittest discover -s tests -p 'test*.py'
python examples/run_toy.py --out runs/toy-check
```

Coverage includes admission grouping, grounded templates, frozen text features
with a shared trainable projection, Soft/Hard/Skip behavior, paired rewards,
short-trajectory REINFORCE, two-stage training, checkpoint recovery, validation
selection and threshold reuse, and held-out inference. CI also builds the wheel
and checks packaged pool resources outside the source checkout.

These checks establish software behavior. Real-data training, clinical metrics,
and full frozen-pool validation still require the actual datasets and resources.
