import random

def generate_rows(n):
    print("states,transitions,atomic_props,base_props,refined_props,num_classes,chain_states,bit_width")
    for _ in range(n):
        states = random.choice([1000, 2000, 4000, 8000, 16000])
        transitions = int(states * 0.8)  # keep same proportion
        atomic_props = random.randint(20, 40)
        base_props = random.randint(2, 5)
        refined_props = random.randint(1, 40)
        num_classes = random.randint(10, atomic_props)
        chain_states = random.randint(1, 40)
        bit_width = random.randint(5,12)  # fixed
        
        print(f"{states},{transitions},{atomic_props},{base_props},{refined_props},{num_classes},{chain_states},{bit_width}")

generate_rows(40)