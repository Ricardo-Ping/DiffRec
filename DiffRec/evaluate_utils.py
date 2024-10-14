import numpy as np
import bottleneck as bn
import torch
import math


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))

    return precision, recall, NDCG, MRR


def print_results(loss, valid_result, test_result, topN):
    """输出评估结果，并显示对应的topN值，保留5位小数。"""
    if loss is not None:
        print("[Train]: loss: {:.5f}".format(loss))

    if valid_result is not None:
        print("[Valid]:")
        for idx, n in enumerate(topN):
            print(f"Top-{n}: Precision: {valid_result[0][idx]:.5f} Recall: {valid_result[1][idx]:.5f} "
                  f"NDCG: {valid_result[2][idx]:.5f} MRR: {valid_result[3][idx]:.5f}")

    if test_result is not None:
        print("[Test]:")
        for idx, n in enumerate(topN):
            print(f"Top-{n}: Precision: {test_result[0][idx]:.5f} Recall: {test_result[1][idx]:.5f} "
                  f"NDCG: {test_result[2][idx]:.5f} MRR: {test_result[3][idx]:.5f}")


