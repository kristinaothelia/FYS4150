    




    # easy to use in pytest later
def something():
    #plan to make a function and pytest on this:
    #distances = [1, 1, 1, 1, 1, 1, 2, 4, 3, 7, 0.3, 6, 7, 8, 9, 10] # length 16
    #distances = distances[-9:]     # [4, 3, 7, 0.3, 6, 7, 8, 9, 10] # length 9

    # looping backwards to find the last minimum
    index = int(len(distances)-2)
    while distances[index] < distances[index + 1]:
        #print('index', index);print(distances[index]);print(distances[index+1])
        index -= 1
    pass