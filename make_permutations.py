def make_permutations(n):
    """Generate 10 basis vectors with non-random patterns that can extend to arbitrary lengths.
    
    Args:
        n (int): Length of binary strings (should be multiple of 10 for best results)
        
    Returns:
        list: Array of 10 basis vectors with deterministic patterns
    """
    def create_basis_pattern(pattern_id, length):
        """Create a specific basis pattern for given length."""
        if pattern_id == 0:  # Alternating: 1010101010...
            return ''.join(['1' if i % 2 == 0 else '0' for i in range(length)])
        elif pattern_id == 1:  # Alternating reverse: 0101010101...
            return ''.join(['0' if i % 2 == 0 else '1' for i in range(length)])
        elif pattern_id == 2:  # First half 1s, second half 0s: 1111100000...
            half = length // 2
            return '1' * half + '0' * (length - half)
        elif pattern_id == 3:  # First half 0s, second half 1s: 0000011111...
            half = length // 2
            return '0' * half + '1' * (length - half)
        elif pattern_id == 4:  # Every 4th: 0011001100...
            return ''.join(['1' if (i // 2) % 2 == 1 else '0' for i in range(length)])
        elif pattern_id == 5:  # Every 4th reverse: 1100110011...
            return ''.join(['0' if (i // 2) % 2 == 1 else '1' for i in range(length)])
        elif pattern_id == 6:  # Every 3rd: 001001001...
            return ''.join(['1' if i % 3 == 2 else '0' for i in range(length)])
        # elif pattern_id == 7:  # Fibonacci-like: 0110100110...
        #     pattern = '01101001'
        #     return (pattern * ((length // len(pattern)) + 1))[:length]
        elif pattern_id == 7:  # Quarters: 0011110000...
            quarter = length // 4
            return '0' * quarter + '1' * (2 * quarter) + '0' * (length - 3 * quarter)
        elif pattern_id == 8:  # Every 5th: 00001000010...
            return ''.join(['1' if i % 5 == 4 else '0' for i in range(length)])
        elif pattern_id == 9:  # Block pattern: 000111000111...
            return ''.join(['1' if (i // 3) % 2 == 1 else '0' for i in range(length)])
        # elif pattern_id == 7:  # Fibonacci-like: 0110100110...
        #     pattern = '01101001'
        #     return (pattern * ((length // len(pattern)) + 1))[:length]
        # elif pattern_id == 9:  # Prime positions (approximation): positions 2,3,5,7,11,13...
        #     primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        #     result = ['0'] * length
        #     for p in primes:
        #         if p < length:
        #             result[p] = '1'
        #     return ''.join(result)
        
    result = []
    for i in range(10):
        result.append(create_basis_pattern(i, n))
    return result

if __name__ == "__main__":
    # Test with n=10
    print(make_permutations(10))