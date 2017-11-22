assignments = []


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


rows = 'ABCDEFGHI'
cols = '123456789'


def cross(set_a, set_b):
    "Cross product of elements in A and elements in B."
    return [s + t for s in set_a for t in set_b]


boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI')
                for cs in ('123', '456', '789')]
diagonal_units = [
    [a + b for a, b in zip(rows, cols)], [a + b for a, b in zip(rows[::-1], cols)]]  # Get diagonals
unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    for box, value in values.items():
        if len(value) == 2:  # Look for a twin
            for p_box in peers[box]:
                p_value = values[p_box]
                # Found it, now remove from peers
                if len(p_value) == 2 and ((value[0] == p_value[0] and value[1] == p_value[1]) or (value[0] == p_value[1] and value[1] == p_value[0])):
                    for p_p_box in peers[box] & peers[p_box]:
                        values = assign_value(values, p_p_box, values[p_p_box].replace(
                            value[0], '').replace(value[1], ''))
                    break  # Can't have two twins, so break
    return values


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[s]) for s in boxes)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF':
            print(line)
    return


def eliminate(values):
    """Eliminate values using the elimination strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values after applying elimination.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values = assign_value(
                values, peer, values[peer].replace(digit, ''))
    return values


def only_choice(values):
    """Eliminate values using the only choice strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary after applying the only choice strategy.
    """
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values = assign_value(values, dplaces[0], digit)
    return values


def reduce_puzzle(values):
    """Eliminate values using all strategies repeatedly until we stall.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the solved or stalled values dictionary or "False" if the sudoku is unsolvable.
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len(
            [box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        solved_values_after = len(
            [box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Solve stalled sudokus by applying recursive Depth First Search.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        The solved sudoku, or "False" if unsolvable.
    """
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return values
    if all(len(values[s]) == 1 for s in boxes):
        return values  # Solved!

    # Choose one of the unfilled squares with the fewest possibilities
    n, easiest_box = min((len(values[s]), s)
                         for s in boxes if len(values[s]) > 1)

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for val in values[easiest_box]:
        new_values = values.copy()
        new_values = assign_value(new_values, easiest_box, val)
        slvd_new_values = search(new_values)
        if slvd_new_values:
            return slvd_new_values

    return False


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    return search(grid_values(grid))


if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
