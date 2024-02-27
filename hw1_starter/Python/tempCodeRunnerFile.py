    def threshold(v):
        u = np.zeros(v.shape)  # initialize
        u = v * (v>0).astype(float)
        return u

    v = np.array([3, 5, -2, 5, -1, 0])
    print(threshold(v))  # call from command line