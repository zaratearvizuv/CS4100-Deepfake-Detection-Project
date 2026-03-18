import datasets

ossia = datasets.load_dataset("ductai199x/open-set-synth-img-attribution", "arch") # or "gen"

# see structure
print(ossia)

# access the first example
print(ossia["train"][0])
