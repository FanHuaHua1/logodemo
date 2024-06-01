package clustering.bmutils;

public class MetricUtils {
	public static double euclidean(double[] p1, double[] p2) {
		double dist = 0.0;
		for (int i = 0; i < p1.length; i++) {
			double d = p1[i] - p2[i];
			dist += d * d;
		}
		return Math.sqrt(dist);
	}
}
