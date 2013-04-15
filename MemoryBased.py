#!/usr/bin/python

import sys
import math
import numpy as np

# Computes the average rating provided by the user
def computeAverageRating(ratingArr, numMovies):
    totRating = 0
    numRating = 0
    for i in xrange(numMovies):
        if(ratingArr[i] > 0):
            totRating += ratingArr[i]
            numRating += 1
    return 1.0 * totRating / numRating

# Computes the variance rating provided by the user. This is not normalized by the number of movies that have been rated
def computeVariance(ratingArr1, ratingArr2, numMovies):
    totResult = 0
    avgRating = computeAverageRating(ratingArr1, numMovies)
    for i in xrange(numMovies):
        if(ratingArr1[i] > 0 and ratingArr2[i] > 0):
            totResult += pow((ratingArr1[i] - avgRating), 2)
    return totResult

# Compute the Pearson's Correlation Similarity
def computePearsonCorrelationSimilarity(ratingArr1, ratingArr2, numMovies):
    result      = 0
    u1StdDev    = np.sqrt(computeVariance(ratingArr1, ratingArr2, numMovies))
    u2StdDev    = np.sqrt(computeVariance(ratingArr2, ratingArr1, numMovies))
    u1AvgRating = computeAverageRating(ratingArr1, numMovies)
    u2AvgRating = computeAverageRating(ratingArr2, numMovies)
    for i in xrange(numMovies):
        if(ratingArr1[i] > 0 and ratingArr2[i] > 0):
            result += (ratingArr1[i] - u1AvgRating) * (ratingArr2[i] - u2AvgRating)
    result = result / (u1StdDev * u2StdDev) if (result > 0 and u1StdDev > 0 and u2StdDev > 0) else 0
    return result

# Computes the modulus of a vector
def computeVectorModulus(vector, dimension):
    result = 0
    for i in xrange(dimension):
        if(vector[i] > 0):
            result += vector[i] ** 2
    return np.sqrt(result)

# Computes the Cosine similarity between the two users
def computeCosineSimilarity(vector1, vector2, dimension):
    result = 0
    u1Mod = computeVectorModulus(vector1, dimension)
    u2Mod = computeVectorModulus(vector2, dimension)
    for i in xrange(dimension):
        if(vector1[i] > 0 and vector2[i] > 0):
            result += vector1[i] * vector2[i]
    result = result / (u1Mod * u2Mod) if (u1Mod > 0 and u2Mod > 0) else result
    return result

# Computes the adjusted cosine similarity between movie1 and movie2
def computeAdjustedCosineSimilarity(ratingMatrix, movie1, movie2, normalizeFactor1, normalizeFactor2, uAvgRatingArr, numUsers):
    normalizeFactor1 = 0
    normalizeFactor2 = 0
    result           = 0
    for i in xrange(numUsers):
        result += (ratingMatrix[i][movie1] - uAvgRatingArr[i]) * (ratingMatrix[i][movie2] - uAvgRatingArr[i])
    result = result / normalizeFactor1 * normalizeFactor2 if (normalizeFactor1 > 0 and normalizeFactor2 > 0) else result
    return result

# Computes the Inverse User Frequency metric
def computeInverseUserFreq(trainingMatrix, numTrainingUsers, numMovies):
    inverseFreq = [0 for i in xrange(numMovies)]
    for i in xrange(numMovies):
        numUsersVoted = 0
        for j in xrange(numTrainingUsers):
            if(trainingMatrix[j][i] > 0):
                numUsersVoted += 1
        ratio = 1.0 * numTrainingUsers / numUsersVoted if numUsersVoted > 0 else numTrainingUsers
        inverseFreq[i] = np.log(ratio)
    return inverseFreq  

# Computes the modulus of a vector taking Inverse User Frequency into account
def computeVectorModulusIUF(vector, dimension, inverseFreq):
    result = 0
    for i in xrange(dimension):
        if(vector[i] > 0):
            result += (vector[i] * inverseFreq[i]) ** 2
    return np.sqrt(result)

# Computes the Cosine similarity between the two users taking Inverse User Frequency into account
def computeCosineSimilarityIUF(vector1, vector2, dimension, inverseFreq):
    result = 0
    u1Mod = computeVectorModulusIUF(vector1, dimension, inverseFreq)
    u2Mod = computeVectorModulusIUF(vector2, dimension, inverseFreq)
    for i in xrange(dimension):
        if(vector1[i] > 0 and vector2[i] > 0):
            result += vector1[i] * vector2[i] * inverseFreq[i]
    result = result / (u1Mod * u2Mod) if (u1Mod > 0 and u2Mod > 0) else result
    return result

# Fetch the ith column of a matrix
def getColumn(matrix, i):
    return [row[i] for row in matrix]

# Compute the movie similarity matrix 
def computeMovieSimilarity(ratingMatrix, numUsers, numMovies):
    similarMat    = [[1 for i in xrange(numMovies)] for j in xrange(numMovies)]
    for i in xrange(numMovies):
        for j in xrange(numUsers):
            normalizeFactor[i] += (ratingMatrix[j][i] - uAvgRatingArr[j]) ** 2
        normalizeFactor[i] = np.sqrt(normalizeFactor[i])
    for i in xrange(numMovies):
        ratingArr1 = getColumn(ratingMatrix, i)
        for j in xrange(i + 1, numMovies):
            ratingArr2       = getColumn(ratingMatrix, j)
            similarity       = computeCosineSimilarity(ratingArr1, ratingArr2, numUsers)
            similarMat[i][j] = similarity
            similarMat[j][i] = similarity
    return similarMat

# Predicts the rating a user will give to a movie given the test user and movieId for Item based algorithm
def predictRatingItemBasedCF(ratingArr, similarArr, numMovies):
    result = 0
    simMod = 0
    for i in xrange(numMovies):
        if(ratingArr[i] > 0):
            result += similarArr[i] * ratingArr[i]
            simMod += math.fabs(similarArr[i])
    result = 1.0 * result / simMod if simMod > 0 else result
    return result

# Predicts the rating a user will give to a movie given the test user and movieId
def predictRating(ratingArr, trainingMatrix, numTrainingUsers, movieId, numMovies, algorithm, inverseFreq):
    result     = 0
    weightSum  = 0
    uAvgRating = computeAverageRating(ratingArr, numMovies)
    for i in xrange(numTrainingUsers):
        if(trainingMatrix[i][movieId] > 0):
            if(algorithm == 1):
                weight = computePearsonCorrelationSimilarity(ratingArr, trainingMatrix[i], numMovies)
            elif(algorithm == 2):
                weight = computeCosineSimilarity(ratingArr, trainingMatrix[i], numMovies)
            elif(algorithm == 3):
                weight = computeCosineSimilarityIUF(ratingArr, trainingMatrix[i], numMovies, inverseFreq)
            weightSum += math.fabs(weight)
            result    += weight * (trainingMatrix[i][movieId] - uAvgRating)
    result = uAvgRating + result / weightSum if weightSum > 0 else uAvgRating
    return result

# Computes the rating for all test users and movies that need a prediction of rating
def predictRatingForAllUsers(trainingMatrix, filePath, numTrainingUsers, numMovies, numTestUsers, algorithm, offset):
    testFile   = open(filePath, "r")
    testMatrix = [[-1 for i in xrange(numMovies)] for j in xrange(numTestUsers)]
    for line in testFile:
        testArr = line.split(' ')
        userId  = int(testArr[0])
        movieId = int(testArr[1]) - 1
        rating  = int(testArr[2])
        if(userId > numTrainingUsers):
            testMatrix[userId - offset][movieId] = rating
    testFile.close()
    testOutput  = open("result.txt", "w")
    inverseFreq = computeInverseUserFreq(trainingMatrix, numTrainingUsers, numMovies) if algorithm == 3 else None
    if(algorithm > 0 and algorithm < 4):
        for i in range(numTestUsers):
            for j in range(numMovies):
                if(testMatrix[i][j] == 0):
                    prediction = predictRating(testMatrix[i], trainingMatrix, numTrainingUsers, j, numMovies, algorithm, inverseFreq)
                    prediction = math.floor(prediction + 0.5)
                    testOutput.write(`i + offset` + " " + `j + 1` + " " + `prediction` + "\n")
    elif(algorithm == 4):
        similarMat = computeMovieSimilarity(trainingMatrix, numTrainingUsers, numMovies)
        for i in xrange(numTestUsers):
            for j in xrange(numMovies):
                if(testMatrix[i][j] == 0):
                    prediction = predictRatingItemBasedCF(testMatrix[i], similarMat[j], numMovies)
                    prediction = math.floor(prediction + 0.5)
                    prediction = 1.0 if prediction < 1.0 else prediction
                    prediction = 5.0 if prediction > 5.0 else prediction
                    testOutput.write(`i + offset` + " " + `j + 1` + " " + `prediction` + "\n")
    testOutput.close()

def main(argv):
    trainingFile, testFile, numUsers, numMovies, numTestUsers, algorithm, offset = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
    trainingMatrix = np.fromfile(trainingFile, sep = '\n', dtype = 'int').reshape(numUsers, numMovies)
    predictRatingForAllUsers(trainingMatrix, testFile, numUsers, numMovies, numTestUsers, algorithm, offset)

if(__name__ == '__main__'):
    sys.exit(main(sys.argv))
