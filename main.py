words = [
    "selling",
    "Yes",
    "Would",
    "Qual",
    "PORT",
    "No",
    "added",
    "Low",
    "NO",
    "High",
    "Few",
    "quantity",
    "Compare",
    "expire",
    "IGH",
    "Last",
    "Limited",
    "demand",
    "No",
    "risk",
    "limited",
    "sell",
    "ULAR",
    "purchased",
    "Already",
    "OST",
    "Only",
    "elling",
    "Only",
    "bought",
    "DEM",
    "Oklahoma",
    "NO",
    "LOG",
    "Sold",
    "Less",
    "withdraw",
    "Few",
    "Quantity",
    "OULD",
    "Quantity",
    "REG",
    "dont",
    "Limited",
    "TX",
    "remaining",
    "ary",
    "INS",
    "ess",
    "Tool,",
]
scores = [
    0.76887405,
    0.6648371,
    0.65965315,
    0.59787788,
    0.57609911,
    0.57062238,
    0.55737576,
    0.52763443,
    0.49998377,
    0.49994594,
    0.49993744,
    0.49990535,
    0.49862194,
    0.49590333,
    0.49564061,
    0.48954947,
    0.47959636,
    0.47690435,
    0.47473223,
    0.46749409,
    0.46222366,
    0.45059009,
    0.44686411,
    0.43657307,
    0.42663975,
    0.42152317,
    0.4192521,
    0.41571051,
    0.41206567,
    0.40943102,
    0.40912457,
    0.39385651,
    0.39341388,
    0.39009115,
    0.38565691,
    0.38368293,
    0.37541005,
    0.37513233,
    0.37396742,
    0.36179707,
    0.36130326,
    0.36068768,
    0.35766445,
    0.35417648,
    0.3532743,
    0.34498724,
    0.34282246,
    0.34259678,
    0.34123984,
    0.34086646,
]

tgts = {
    "selling",
    "Yes",
    "Would",
    "PORT",
    "No",
    "added",
    "Low",
    "High",
    "Few",
    "quantity",
    "Compare",
    "expire",
    "Last",
    "Limited",
    "demand",
    "risk",
    "sell",
    "purchased",
    "Already",
    "Only",
    "bought",
    "Sold",
    "Less",
    "withdraw",
    "remaining",
}

print(len(tgts))

occured = {w: False for w in tgts}

for w, s in zip(words, scores):
    if w in tgts and occured[w] is False:
        print(f"{w}, {round(s, 3):.3f}")
        occured[w] = True
