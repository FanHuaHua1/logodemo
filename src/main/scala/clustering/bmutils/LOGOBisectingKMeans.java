package clustering.bmutils;

import smile.clustering.KMeans;
import java.io.Serializable;
import static clustering.bmutils.MetricUtils.euclidean;
public class LOGOBisectingKMeans implements Serializable {

	private final int k;
	private final int m; // times of bisecting trials
	public double[][] centroids;

	public LOGOBisectingKMeans(int k, int m) {
		this.k = k;
		this.m = m;
	}

	public int predict(double[] x) {
		double nearest = Double.MAX_VALUE;
		int label = 0;
		for (int j = 0; j < k; j++) {
			double dist = euclidean(this.centroids[j], x);
			if (dist < nearest) {
				nearest = dist;
				label = j;
			}
		}
		return label;
	}

	private double computeSSE(double[][] centroids, double[][] Points) {
		double sse = 0.0;
		for (double[] point : Points) {
			double d1 = euclidean(point, centroids[0]);
			double d2 = euclidean(point, centroids[1]);
			double distance = Math.min(d1, d2);
			sse += distance * distance;
		}
		return sse;
	}

	public void clustering(double[][] features) {
		KMeans bestBisectingKmeans = null;
		double minTotalSSE = Double.MAX_VALUE;
		for (int i = 0; i < m; i++) {// 试验执行m次kmeans
			// smile的kmeans
			final KMeans smileModel = KMeans.fit(features, k, 5,1E-5);
			// 计算SSE
			double currentTotalSSE = computeSSE(smileModel.centroids, features);// 计算一次二分试验中总的SSE的值
			if(bestBisectingKmeans == null) {   //第一次迭代
				bestBisectingKmeans = smileModel;
				minTotalSSE = currentTotalSSE;
			} else {
				if(currentTotalSSE < minTotalSSE) {
					bestBisectingKmeans = smileModel;  // 记录总SSE最小的二分聚类，通过kmeans保存二分结果
					minTotalSSE = currentTotalSSE;
				}
			}
		}
		this.centroids = bestBisectingKmeans.centroids;
	}

}
