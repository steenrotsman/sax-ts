from sax_ts import sax

from tests.reference import discretise

data = [[-1.0, 2.0, 0.1, -1.0, 1.0, -1.0], [1.0, 3.2, -1.0, -3.0, 1.0, -1.0]]
for row in data:
    print(sax(row, 6, 1, 3, 2))
    print(discretise(row, 6, 1, 3, 2))
