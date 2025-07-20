def make_permutations(n):
    """Generate array of binary strings with exactly one '1' at each position.
    
    Args:
        n (int): Length of binary strings
        
    Returns:
        list: Array of strings where each has one '1' and rest '0's
    """
    result = []
    for i in range(n):
        binary_string = '0' * n
        binary_string = binary_string[:i] + '1' + binary_string[i+1:]
        result.append(binary_string)
    return result

if __name__ == "__main__":
    # Test with n=3
    print(make_permutations(3))