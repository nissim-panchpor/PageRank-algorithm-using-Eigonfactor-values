"""
This python script is a slight variation of the PageRank algorithm, which uses Eigonfactor values for ranking nodes/pages. 

Input:
Takes an edge list as its input. An edge list is a csv file containing 3 columns - from node, to node and weight for edges (or number of citations)
for 'from node' to the 'to node'.
The default input is an edge list containing 4 million edges amongst 11 thousand nodes.

Output:
Produces the top 20 pages by popularity in descending order and corresponding eigonfactor values.

"""


if __name__ == "__main__":
    # execute only if run as a script
    print("This is run as a script, not a module!")

    import numpy as np
    import pandas as pd
    import math as m
    import sys
    from sklearn import preprocessing as pp
    import timeit
    #import networkx as nx


    def load_data(file):
        """
        Read links dataset into a dataframe
        """

        # Read the csv dataset into a pandas dataframe
        links_df = pd.read_csv(file, sep = ",", header=None)

        # Convert the dataframe into a matrix
        links_data = links_df.pivot(index=1, columns=0, values=2)

        # Get the rows and columns of the above matrix
        columns = links_data.columns.values
        indices = links_data.index.values

        # adjacency matrix is a square matrix, hence need to add 0 columns if dimensions of matrix are not same
        if links_data.shape[0] != links_data.shape[1]:
            missing_col = 0
            for i in range(max(links_data.shape)):
                if i not in columns:
                    missing_col += 1
                    links_data[i] = np.zeros(max(links_data.shape), dtype=int)
            
            # Do the same for index
            missing_ind = 0
            for i in range(max(links_data.shape)):
                if i not in indices:
                    missing_ind += 1
                    print(i)

        return links_data


    def get_eigenfactor(Z):
        """
        Processes the input adjacency matrix to calculate eigonfactors
        """

        # Start the clock
        start_time = timeit.default_timer()
       
        # Set diagonal values to 0
        np.fill_diagonal(Z, 0)
        #Z = np.nan_to_num(Z)
        
        # Normalize the Z matrix to get H matrix
        
        H = pp.normalize(Z, axis=0, norm="l1")
        
        #Find the dangling nodes
        d = np.sum(H, axis=0) == 0
        d = d.astype(int)
        
        #Initialize the constants
        alpha = 0.85
        epsilon = 0.00001

        # Initialize article Vector 'a' depending on example or big dataset
        if(H.shape == (6, 6)):
            number = np.array([3, 2, 5, 1, 2, 1])
        else:
            number = np.ones(H.shape[1])

        A_tot = sum(number)
        A = number/A_tot
        A = A.reshape(A.shape[0], 1)

        # Create initial influence vector
        n = H.shape[1]
        pizero = np.ones(H.shape[1])/n
        pizero = pizero.reshape(pizero.shape[0], 1)


        # Calculate Influence Vector
        piprev = pizero
        iteration = 0
    
        while(True):
            iteration += 1
            pinext = alpha * np.dot(H, piprev) + (alpha * np.dot(d, piprev) + (1 - alpha)) * A

            # Find the residual by taken absolute difference of current and previous influence vectors
            residual_sum = np.absolute(pinext - piprev).sum()

            if(residual_sum < epsilon):
                print("Ok, Done with the loop!")
                break
            else:
                piprev = pinext
                print("Iterating for ", iteration)
                sys.stdout.flush()

        # Find eigenfactor matrix
        eigonfactor = 100 * (np.dot(H, pinext) / sum(np.dot(H, pinext)))

        # Loop done, find the time taken
        end_time = timeit.default_timer()
        run_time = end_time - start_time

        return eigonfactor, iteration, run_time


############     Execution of the program begins from this part    ##############

    #Provide the links dataset
    fname = "links.txt"

    # Load the links dataset into a a data frame
    links_data = load_data(fname)
    #print(type(links_data))
    columns = links_data.columns.values
    indices = links_data.index.values

    # Sort the column values in order and reassign
    columns = np.sort(columns)
    links_data = links_data[list(columns)]

    # Replace NaN values by 0
    Z = np.nan_to_num(np.array(links_data))

    # Process the example matrix

    print("\nProcessing example Z matrix")

    # Initialize example Z matrix
    Z_example = np.array([[1, 0, 2, 0, 4, 3],
                    [3, 0, 1, 1, 0, 0],
                    [2, 0, 4, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1],
                    [8, 0, 3, 0, 5, 2],
                    [0, 0, 0, 0, 0, 0]])

    print("\nExample Z matrix is :\n", Z_example)

    Z_example_col = ["A", "B", "C", "D", "E", "F"]
    Z_example_ind = Z_example_col

    # Call get_eigenfactor() method to find eigonfactors for example matrix
    eigonfactor, iterations, run_time = get_eigenfactor(Z_example)
    #print(eigenfactor)
    
    # Create a dataframe from eigonvector output above
    eigonfactor = pd.DataFrame(data=eigonfactor, index=Z_example_col, columns=["eigonvalues"])

    print("\n\nEigonfactor Z matrix for example is :\n", eigonfactor)

    print("\nBegin processing links.txt network")

    # Call get_eigenfactor() method to find eigonfactors for links.txt datatset
    eigonfactor, iterations, run_time = get_eigenfactor(Z)

    # Create a dataframe from eigonvector output above
    eigonfactor = pd.DataFrame(data=eigonfactor, index=columns, columns=["eigonvalues"])
    #print(eigenfactor)

    eigonfactor_sorted = eigonfactor.sort_values(by="eigonvalues", ascending=False)

    ######    Print final answers      ######

    print("\nTop 20 pages and their eigonfactor values are: \n\n", eigonfactor_sorted[:20])
    print("\nNumber of iterations taken:", iterations)
    print("\nTime taken to find eigonfactors for links dataset:", run_time, "seconds")



################### OUTPUT for example Z matrix #########################

# Eigonfactor Z matrix for example is :
#     eigonvalues
# A    34.051006
# B    17.203742
# C    12.175455
# D     3.653164
# E    32.916632
# F     0.000000

# Number of iterations taken: 17


################### OUTPUT for links dataset #########################
# 
# Top 20 pages and their eigonfactor values are:

#        eigonvalues
# 4408     1.448119
# 4801     1.412719
# 6610     1.235035
# 2056     0.679502
# 6919     0.664879
# 6667     0.634635
# 4024     0.577233
# 6523     0.480815
# 8930     0.477773
# 6857     0.439735
# 5966     0.429718
# 1995     0.386207
# 1935     0.385120
# 3480     0.379578
# 4598     0.372789
# 2880     0.330306
# 3314     0.327508
# 6569     0.319272
# 5035     0.316779
# 1212     0.311257

# Number of iterations taken: 32

# Time taken to find eigonfactors for links dataset: 6.053250215221036 seconds
