import random;
from functools import reduce
def randomPicker(howMany, *ranges):
    mergedRange = reduce(lambda a, b: a + b, ranges);
    ans = [];
    for i in range(howMany):
        ans.append(random.choice(mergedRange));
    return ans;

x = randomPicker(1, list(range(-10, -5)), list(range(0,2)))
print(x)

