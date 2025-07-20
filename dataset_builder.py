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
        num_to_combine = random.randint(1, 4)
        selected_combos = random.sample(combos, num_to_combine)
        
        # OR the selected combinations
        or_result = ['0'] * combo_length
        for combo in selected_combos:
            for i, bit in enumerate(combo):
                if bit == '1':
                    or_result[i] = '1'
        
        result.append(''.join(or_result))
    
    return result

def dataset_2d(combos, n):
    """Generate n randomly sampled 2D OR combinations from the given combos.
    
    Args:
        combos (list): Array of binary strings from make_permutations
        n (int): Number of samples to generate
        
    Returns:
        list: Array of n 2D arrays representing OR combinations
    """
    if not combos:
        return []
    
    result = []
    combo_length = len(combos[0])
    
    for _ in range(n):
        # Randomly sample combinations to OR together
        num_to_combine = random.randint(1, 4)
        selected_combos = random.sample(combos, num_to_combine)
        
        # OR the selected combinations into 2D array
        or_result = [[0 for _ in range(combo_length)] for _ in range(combo_length)]
        
        for combo in selected_combos:
            for i, bit in enumerate(combo):
                if bit == '1':
                    for j in range(combo_length):
                        or_result[i][j] = 1
                        or_result[j][i] = 1
        
        result.append(or_result)
    
    return result

if __name__ == "__main__":
    # Test with permutations of length 4
    combos = make_permutations(10)
    print("Original combos:", combos)
    
    # Generate 5 random OR combinations
    dataset = basic_dataset(combos, 5)
    print("Generated dataset:", dataset)
    
    # Generate 2 random 2D OR combinations
    # dataset_2d_result = dataset_2d(combos, 2)
    # print("Generated 2D dataset:")
    # for i, matrix in enumerate(dataset_2d_result):
    #     print(f"Matrix {i+1}:")
    #     for row in matrix:
    #         print(row)
