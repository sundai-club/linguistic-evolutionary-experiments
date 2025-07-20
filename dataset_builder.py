import random
from make_permutations import make_permutations

def basic_dataset(combos, n):
    """Generate n randomly sampled OR combinations from the given combos.
    
    Args:
        combos (list): Array of binary strings from make_permutations
        n (int): Number of samples to generate
        
    Returns:
        list: Array of n binary strings representing OR combinations
    """
    if not combos:
        return []
    
    result = []
    combo_length = len(combos[0])
    
    for _ in range(n):
        # Randomly sample combinations to OR together
        num_to_combine = random.randint(1, len(combos))
        selected_combos = random.sample(combos, num_to_combine)
        
        # OR the selected combinations
        or_result = ['0'] * combo_length
        for combo in selected_combos:
            for i, bit in enumerate(combo):
                if bit == '1':
                    or_result[i] = '1'
        
        result.append(''.join(or_result))
    
    return result

if __name__ == "__main__":
    # Test with permutations of length 4
    combos = make_permutations(4)
    print("Original combos:", combos)
    
    # Generate 5 random OR combinations
    dataset = basic_dataset(combos, 5)
    print("Generated dataset:", dataset)