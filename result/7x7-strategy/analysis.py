file_name = "dqn_mal.nohup.out"

with open(file_name) as f:
    line = f.readline()
    while line:
        if "throughput" in line:
            print(line)
        line = f.readline()

if __name__ == "__main__":
    pass