package gr.iti.mklab.multimedia.webservice;

import gr.iti.mklab.visual.aggregation.VladAggregatorMultipleVocabularies;
import gr.iti.mklab.visual.datastructures.Linear;
import gr.iti.mklab.visual.dimreduction.PCA;
import gr.iti.mklab.visual.extraction.AbstractFeatureExtractor;
import gr.iti.mklab.visual.extraction.SURFExtractor;
import gr.iti.mklab.visual.utilities.Answer;
import gr.iti.mklab.visual.vectorization.ImageVectorization;
import gr.iti.mklab.visual.vectorization.ImageVectorizationResult;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;

import javax.imageio.ImageIO;

public class Test {

	private static String[] urls = {
		"http://pbs.twimg.com/media/BqPlbtaCEAAr2o5.png",
		"http://pbs.twimg.com/media/BqW8HD-CYAAOloZ.jpg",
		"http://pbs.twimg.com/media/BqZ0gAFIUAAmwXp.png",
		"http://pbs.twimg.com/media/BqZlzqrIYAEKauK.jpg",
		"https://pbs.twimg.com/media/BqVs2rnCAAEnH0r.jpg:large",
		"http://pbs.twimg.com/media/BqXBRYgCcAILG13.jpg",
		"http://pbs.twimg.com/media/BqWyM0JCMAAkvq4.jpg"
	};
	
	private static int targetLengthMax = 1024;
	private static int targetLength = 1024;
	
	private static int maxNumPixels = 768 * 512;
	private static int[] numCentroids = {128, 128, 128, 128};
	
	public static void main(String...args) throws Exception {
		
		File learningFolder = new File("/disk2_data/VisualIndex/learning_files");
		
		String[] codebookFiles = { 
				learningFolder.toString() + "/surf_l2_128c_0.csv",
				learningFolder.toString() + "/surf_l2_128c_1.csv", 
				learningFolder.toString() + "/surf_l2_128c_2.csv",
				learningFolder.toString() + "/surf_l2_128c_3.csv" 
		};
		
		File pcaFile = new File(learningFolder, "pca_surf_4x128_32768to1024.txt");

		VladAggregatorMultipleVocabularies vladAggregator = 
				new VladAggregatorMultipleVocabularies(codebookFiles, numCentroids, AbstractFeatureExtractor.SURFLength);
	
		ImageVectorization.setFeatureExtractor(new SURFExtractor());
		ImageVectorization.setVladAggregator(vladAggregator);
		
		int initialLength = numCentroids.length * numCentroids[0] * AbstractFeatureExtractor.SURFLength;
		
		if(targetLength < initialLength) {
			PCA pca = new PCA(targetLength, 1, initialLength, true);
			pca.loadPCAFromFile(pcaFile.toString());
			ImageVectorization.setPcaProjector(pca);
		}
		
		Linear linearIndex = new Linear(targetLengthMax, 100, false, "/disk1_data/workspace/git/ieee_mag/images", true, true, 0);
		System.out.println("Index Size: " + linearIndex.getLoadCounter());
		
		for(String urlStr : urls) {
			URL url = new URL(urlStr);  	
			BufferedImage image = ImageIO.read(url.openStream());

			System.out.println(urlStr);
			System.out.println(image.getWidth() + " x " + image.getHeight());
    	
			ImageVectorization imvec = new ImageVectorization(null, image, targetLengthMax, maxNumPixels);
		
			ImageVectorizationResult imvr = imvec.call();
			double[] vector = imvr.getImageVector();

			System.out.println("Vector:" + vector.length);
			
			//String id = urlStr.substring(urlStr.indexOf("/"), urlStr.length()-3);
			
			//linearIndex.indexVector(id, vector);
			
			Answer answer = linearIndex.computeNearestNeighbors(5 * 1, vector);
			System.out.println(answer.getResults());
			System.out.println("============================");
		}
		
		

	}
}
