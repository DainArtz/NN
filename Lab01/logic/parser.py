def _parse_input_line(line: str) -> tuple[list[float], float]:
    line = line.rstrip()
    tokens = line.split(" ")

    try:
        assert len(tokens) == 7
        assert tokens[0].endswith("x")
        assert tokens[1] in ["+", "-"]
        assert tokens[2].endswith("y")
        assert tokens[3] in ["+", "-"]
        assert tokens[4].endswith("z")
        assert tokens[5] == "="

        free_term = float(tokens[6])

    except Exception as e:
        raise Exception(f"Invalid input! - {str(e)}")

    first_coefficient_sign = 1.0
    first_coefficient_str = tokens[0][:-1]
    if first_coefficient_str.startswith("-"):
        first_coefficient_sign = -1.0
        first_coefficient_str = first_coefficient_str[1:]
    first_coefficient = (float(first_coefficient_str) if first_coefficient_str != "" else 1.0) * first_coefficient_sign

    second_coefficient_sign = 1.0
    if tokens[1] == "-":
        second_coefficient_sign = -1.0
    second_coefficient_str = tokens[2][:-1]
    second_coefficient = (float(second_coefficient_str) if second_coefficient_str != ""
                          else 1.0) * second_coefficient_sign

    third_coefficient_sign = 1.0
    if tokens[3] == "-":
        third_coefficient_sign = -1.0
    third_coefficient_str = tokens[4][:-1]
    third_coefficient_str = (float(third_coefficient_str) if third_coefficient_str != ""
                             else 1.0) * third_coefficient_sign

    return [first_coefficient, second_coefficient, third_coefficient_str], free_term


def parse_input(input_path: str) -> tuple[list[list[float]], list[float]]:
    with open(input_path, "rt", encoding="utf-8") as fd:
        lines = fd.readlines()

    assert len(lines) == 3

    coefficients_matrix = []
    free_terms = []

    for line in lines:
        coefficients, free_term = _parse_input_line(line)

        coefficients_matrix.append(coefficients)
        free_terms.append(free_term)

    return coefficients_matrix, free_terms
