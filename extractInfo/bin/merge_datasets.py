import os

def create_merged_dataset(a_loc, b_loc, output_loc):

    assert os.path.isfile(a_loc)
    assert os.path.isfile(b_loc)

    with open(a_loc, 'r') as a:
        data_a = a.read()
    with open(b_loc, 'r') as b:
        data_b = b.read()
    out_data = data_a + data_b
    with open(output_loc, 'w') as output:
        output.write(out_data)

create_merged_dataset("Vault/Datasets/cnn.txt", "Vault/Datasets/numeracy.txt", "Vault/Datasets/combined.txt")
