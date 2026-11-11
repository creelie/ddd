"""Exact finite checks for product-influence.tex.

Run with Python 3. The standard library is sufficient. These checks do
not establish the general theorem or verify the external proof inputs.
"""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


def product_law(alphabets, probabilities):
    points = list(product(*alphabets))
    law = {}
    for point in points:
        probability = F(1)
        for index, value in enumerate(point):
            probability *= probabilities[index][value]
        law[point] = probability
    assert sum(law.values()) == 1
    return points, law


def replacement(point, index, value):
    return point[:index] + (value,) + point[index + 1:]


def increments(values, points, law, probabilities, squared=False):
    answer = []
    for index, marginal in enumerate(probabilities):
        increment = F(0)
        for point in points:
            for value, probability in marginal.items():
                difference = abs(values[point] - values[replacement(point, index, value)])
                if squared:
                    difference *= difference
                increment += law[point] * probability * difference
        answer.append(increment)
    return answer


def variance(values, points, law):
    mean = sum(law[x] * values[x] for x in points)
    return sum(law[x] * (values[x] - mean) ** 2 for x in points)


def boolean_rows(values, points):
    """Choose inexpensive valid rows from a small exact rational grid."""
    size = len(points[0])
    candidates = sorted(product((F(0), F(1, 2), F(1)), repeat=size), key=sum)
    rows = {}
    for point in points:
        for candidate in candidates:
            if all(
                abs(values[point] - values[other])
                <= sum(candidate[i] for i in range(size) if point[i] != other[i])
                for other in points
            ):
                rows[point] = candidate
                break
        assert point in rows
    return rows


def product_boolean_pair(values, rows, points, law, probabilities, weights):
    size = len(weights)
    left = F(0)
    for x in points:
        for y in points:
            if values[x] == 1 and values[y] == 0:
                denominator = sum(
                    weights[i] * rows[x][i] * rows[y][i]
                    for i in range(size) if x[i] != y[i]
                )
                assert denominator > 0
                left += law[x] * law[y] / denominator
    first = increments(values, points, law, probabilities)
    right = sum(first[i] / weights[i] for i in range(size)) / 2
    assert left <= right, (values, probabilities, left, right)
    return left, right


def conditional_cube_average(values, rows, alphabets, probabilities, weights):
    """Evaluate each conditional antipodal inequality before averaging."""
    options = []
    for alphabet, marginal in zip(alphabets, probabilities):
        coordinate_options = []
        for left_index, left in enumerate(alphabet):
            for right in alphabet[left_index:]:
                probability = marginal[left] * marginal[right]
                if left != right:
                    probability *= 2
                if probability:
                    coordinate_options.append((left, right, probability))
        options.append(coordinate_options)

    averaged_left = F(0)
    averaged_right = F(0)
    cube_count = 0
    for choice in product(*options):
        cube_probability = F(1)
        for entry in choice:
            cube_probability *= entry[2]
        varying = [i for i, (left, right, _) in enumerate(choice) if left != right]
        orientations = list(product((0, 1), repeat=len(varying)))
        left_sum = F(0)
        right_sum = F(0)
        for orientation in orientations:
            x = [entry[0] for entry in choice]
            y = list(x)
            for bit, index in zip(orientation, varying):
                x[index] = choice[index][bit]
                y[index] = choice[index][1 - bit]
            x, y = tuple(x), tuple(y)
            if values[x] == 1 and values[y] == 0:
                denominator = sum(weights[i] * rows[x][i] * rows[y][i] for i in varying)
                assert denominator > 0
                left_sum += 1 / denominator
            for index in varying:
                changed = replacement(x, index, y[index])
                right_sum += abs(values[x] - values[changed]) / (2 * weights[index])
        assert left_sum <= right_sum
        averaged_left += cube_probability * left_sum / len(orientations)
        averaged_right += cube_probability * right_sum / len(orientations)
        cube_count += 1
    return averaged_left, averaged_right, cube_count


def check_query_potential(values, alphabets, probabilities):
    points, law = product_law(alphabets, probabilities)
    influence = increments(values, points, law, probabilities, squared=True)
    for chosen in range(len(alphabets)):
        remaining = [i for i in range(len(alphabets)) if i != chosen]
        expected_potential = F(0)
        for value, probability in probabilities[chosen].items():
            if not probability:
                continue
            smaller_alphabets = [alphabets[i] for i in remaining]
            smaller_probabilities = [probabilities[i] for i in remaining]
            smaller_points, smaller_law = product_law(smaller_alphabets, smaller_probabilities)
            restricted = {}
            for point in smaller_points:
                full = point[:chosen] + (value,) + point[chosen:]
                restricted[point] = values[full]
            expected_potential += probability * sum(increments(
                restricted, smaller_points, smaller_law, smaller_probabilities, squared=True
            ))
        assert sum(influence) - expected_potential == influence[chosen]


def check_geometric_derivative(degree):
    """Verify the coefficient calculation modulo z^(2D)+1 over the integers."""
    coefficients = [0] * (2 * degree)
    for index in range(-degree + 1, degree + 1):
        for extra, factor in enumerate((1, -2, 1)):
            exponent = index + degree - 1 + extra
            quotient, remainder = divmod(exponent, 2 * degree)
            coefficients[remainder] += index * factor * ((-1) ** quotient)
    assert coefficients == [2] + [0] * (2 * degree - 1)


def flip_set(point, coordinates):
    return tuple(1 - bit if index in coordinates else bit for index, bit in enumerate(point))


def check_wing_construction(size, selected):
    """Check the matrix identity and find a permutation meeting every budget."""
    vertices = sorted(selected)
    positions = {point: index for index, point in enumerate(vertices)}
    subsets = [tuple(i for i in range(size) if mask & (1 << i)) for mask in range(1 << size)]
    wings = {}
    for point in vertices:
        candidates = [
            subset for subset in subsets
            if all(flip_set(point, smaller) in selected for smaller in subsets if set(smaller) <= set(subset))
        ]
        wings[point] = max(candidates, key=lambda subset: (len(subset), subset))
    weights = [F(index + 1, index + 3) for index in range(size)]

    def monomial(indices):
        answer = F(1)
        for index in indices:
            answer *= weights[index]
        return answer

    agreement = [
        [monomial(i for i in range(size) if x[i] == z[i]) for z in vertices]
        for x in vertices
    ]
    combination = [[F(0) for _ in vertices] for _ in vertices]
    reduced = [[F(0) for _ in vertices] for _ in vertices]
    diagonal = []
    for row, x in enumerate(vertices):
        wing = wings[x]
        factor = F(1)
        for index in wing:
            factor *= 1 - weights[index] ** 2
        diagonal.append(factor)
        for subset in subsets:
            if set(subset) <= set(wing):
                combination[row][positions[flip_set(x, subset)]] = ((-1) ** len(subset)) * monomial(subset)
        for column, z in enumerate(vertices):
            if all(x[index] != z[index] for index in wing):
                reduced[row][column] = agreement[row][column]
    for row in range(len(vertices)):
        for column in range(len(vertices)):
            left = sum(combination[row][middle] * agreement[middle][column] for middle in range(len(vertices)))
            assert left == diagonal[row] * reduced[row][column]

    budgets = [sum(flip_set(x, (index,)) not in selected for x in vertices) for index in range(size)]
    targets = {
        x: [z for z in vertices if all(x[index] != z[index] for index in wings[x])]
        for x in vertices
    }
    ordered = sorted(vertices, key=lambda x: len(targets[x]))

    def assign(level, used, counts):
        if level == len(ordered):
            return True
        x = ordered[level]
        for z in targets[x]:
            if z in used:
                continue
            next_counts = [counts[index] + (x[index] == z[index]) for index in range(size)]
            if any(next_counts[index] > budgets[index] for index in range(size)):
                continue
            if assign(level + 1, used | {z}, next_counts):
                return True
        return False

    assert assign(0, set(), [0] * size)


def main():
    summary = {
        "arithmetic": "exact fractions from Python's standard library",
        "scope": "finite checks supplement the analytic proof",
        "boolean_pair_checks": 0,
        "conditional_orientation_cubes": 0,
        "alphabet_pair_checks": 0,
        "query_potential_checks": 0,
        "linear_equalities": 0,
        "wing_matrix_and_permutation_checks": 0,
        "trigonometric_coefficient_checks": 0,
    }
    for size in (1, 2, 3):
        alphabets = [list(range(2)) for _ in range(size)]
        all_points = list(product(*alphabets))
        laws = [
            [{0: F(1, 2), 1: F(1, 2)} for _ in range(size)],
            [{0: 1 - F(1, 3 + i), 1: F(1, 3 + i)} for i in range(size)],
            [{0: F(1), 1: F(0)} if i == 0 else {0: F(1, 7), 1: F(6, 7)} for i in range(size)],
        ]
        for outputs in product((F(0), F(1)), repeat=len(all_points)):
            values = dict(zip(all_points, outputs))
            rows = boolean_rows(values, all_points)
            for probabilities in laws:
                points, law = product_law(alphabets, probabilities)
                weights = [F(i + 2, i + 1) for i in range(size)]
                direct = product_boolean_pair(values, rows, points, law, probabilities, weights)
                lifted = conditional_cube_average(values, rows, alphabets, probabilities, weights)
                assert direct == lifted[:2]
                summary["boolean_pair_checks"] += 1
                summary["conditional_orientation_cubes"] += lifted[2]
                check_query_potential(values, alphabets, probabilities)
                summary["query_potential_checks"] += size

    alphabets = [list(range(3)), list(range(2))]
    probabilities = [{0: F(1, 1000), 1: F(3, 1000), 2: F(249, 250)}, {0: F(2, 7), 1: F(5, 7)}]
    points, law = product_law(alphabets, probabilities)
    for outputs in product((F(0), F(1)), repeat=len(points)):
        values = dict(zip(points, outputs))
        rows = boolean_rows(values, points)
        weights = [F(2, 5), F(7, 3)]
        direct = product_boolean_pair(values, rows, points, law, probabilities, weights)
        lifted = conditional_cube_average(values, rows, alphabets, probabilities, weights)
        assert direct == lifted[:2]
        summary["alphabet_pair_checks"] += 1
        summary["conditional_orientation_cubes"] += lifted[2]

    for size in range(1, 9):
        alphabets = [list(range(2)) for _ in range(size)]
        probabilities = [{0: F(1, 2), 1: F(1, 2)} for _ in range(size)]
        points, law = product_law(alphabets, probabilities)
        values = {point: F(sum(point), size) for point in points}
        v = variance(values, points, law)
        influence = increments(values, points, law, probabilities, squared=True)
        assert v == F(1, 4 * size)
        assert all(item == 8 * v * v for item in influence)
        summary["linear_equalities"] += 1

    # The certificate proof's final constant is squared to avoid radicals.
    assert (F(4, 60) ** 2) * (F(3, 5) ** 3) == F(15, 125 ** 2)
    summary["certificate_constant_identity"] = "passed exactly after squaring"
    for degree in range(1, 65):
        check_geometric_derivative(degree)
        summary["trigonometric_coefficient_checks"] += 1
    for size in (0, 1, 2, 3):
        points = list(product((0, 1), repeat=size))
        for mask in range(1 << len(points)):
            selected = {point for index, point in enumerate(points) if mask & (1 << index)}
            check_wing_construction(size, selected)
            summary["wing_matrix_and_permutation_checks"] += 1
    summary["status"] = "all checks passed"
    destination = Path(__file__).with_name("exact-check-results.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
