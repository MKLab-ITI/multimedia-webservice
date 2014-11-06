package gr.iti.mklab.multimedia.webservice;

import gr.iti.mklab.multimedia.webservice.exceptions.IndexServiceException;
import gr.iti.mklab.multimedia.webservice.response.JsonResultSet;
import gr.iti.mklab.visual.aggregation.VladAggregatorMultipleVocabularies;
import gr.iti.mklab.visual.datastructures.AbstractSearchStructure;
import gr.iti.mklab.visual.datastructures.IVFPQ;
import gr.iti.mklab.visual.datastructures.Linear;
import gr.iti.mklab.visual.datastructures.PQ;
import gr.iti.mklab.visual.dimreduction.PCA;
import gr.iti.mklab.visual.extraction.AbstractFeatureExtractor;
import gr.iti.mklab.visual.extraction.SURFExtractor;
import gr.iti.mklab.visual.utilities.Answer;
import gr.iti.mklab.visual.utilities.Normalization;
import gr.iti.mklab.visual.utilities.Result;
import gr.iti.mklab.visual.vectorization.ImageVectorization;
import gr.iti.mklab.visual.vectorization.ImageVectorizationResult;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import javax.imageio.ImageIO;
import javax.servlet.ServletContext;
import javax.ws.rs.Consumes;
import javax.ws.rs.DefaultValue;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;
import org.codehaus.jettison.json.JSONArray;

import com.sun.jersey.multipart.FormDataParam;

/**
 * 
 * RESTFUL web-service for visual indexing and nearest neighbor retrieval. 
 * 
 * @author Schinas Manos
 * @email  manosetro@iti.gr
 * 
 */

// Sets the path to base URL + /
@Path("/visual")
public class VisualIndexService {
	
    // hard-coded settings settings
    
    private static final double visualDistanceThreshold = 0.5;
    
    private static int maxNumPixels = 768 * 512;
    
    private static Map<String, AbstractSearchStructure> linearIndices = null;
    private static Map<String, AbstractSearchStructure> ivfpqIndices = null;
    
	private static ServletContext _context;
	
	// Feature extraction
	private static int[] numCentroids = {128, 128, 128, 128};
	private static PCA pca = null;
	
	private static int targetLengthMax = 1024;
	private static int targetLength = 1024;
	
			
	// Product quantization
	private static int numCoarseCentroids = 8192;
	
	private static int subvectorsNum = 64; 
	private static int kc = 256;
	private static int w = 64; // larger values will improve results but increase search time
	
	private static String productQuantizerFile;
	private static String coarseQuantizerFile;

	private static File learningFolder;
	private static File dataFolder;

	private Integer numOfVectors;

	private static DefaultHttpClient httpClient = null;
	
    public VisualIndexService(@Context ServletContext context) throws Exception {
		
    	_context = context;
    	
    	if(httpClient == null) {
    		httpClient = new DefaultHttpClient();
    	}
    	
    	if(learningFolder == null) {    		
    		String learningFolderStr = context.getInitParameter("learningFolder");
    		if(learningFolderStr != null) {
    			learningFolder = new File(learningFolderStr);
    			coarseQuantizerFile = learningFolder + "/" + "qcoarse_1024d_8192k.csv";
    			productQuantizerFile = learningFolder + "/" + "pq_1024_64x8_rp_ivf_8192k.csv";
    		}
    	}
    	
    	if(dataFolder == null) {
    		String dataFolderStr = context.getInitParameter("dataFolder");
    		if(dataFolderStr != null) {
    			dataFolder = new File(dataFolderStr);
    		}
    	}
    	
    	if(pca == null) {
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
    			pca  = new PCA(targetLength, 1, initialLength, true);
				pca.loadPCAFromFile(pcaFile.toString());
				ImageVectorization.setPcaProjector(pca);
    		}
    		
    	}
		
    	if(linearIndices == null) { 
    		linearIndices = new HashMap<String, AbstractSearchStructure>();
    	}
    	if(ivfpqIndices == null){
    		ivfpqIndices = new HashMap<String, AbstractSearchStructure>();
    	}
    	
    	this.numOfVectors = Integer.getInteger(context.getInitParameter("numOfVectors"));
    	if(numOfVectors == null) {
    		numOfVectors = 20000000;
    	}
    	
    }
    
    
    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/qindex/{collection}")
    public String qIndexVector(
    		@PathParam("collection") String collection,
    		@FormDataParam("id") String id,
            @FormDataParam("vector") byte[] bvector,
            @DefaultValue("20") @FormDataParam("numResults") int numResults,
            @DefaultValue("1") @FormDataParam("page") int pageNum,
            @DefaultValue("0") @FormDataParam("threshold") double threshold) {
    	
        // validate parameters
    	try {
    		 if(collection==null || !linearIndices.containsKey(collection) || !ivfpqIndices.containsKey(collection)) {
    			 throw new IndexServiceException(
 						new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
    		 }		 
    		 if (pageNum <= 0) {
				throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "page number < 0"));
    		 }
    		 if (pageNum * numResults > 1000) {
    			 throw new IndexServiceException(
            		new JSONObject().put("code", 400).put("msg", "pageNum * numResults > 1000"));
    		 }
    		 if (bvector.length == 0) {
    			 throw new IndexServiceException(
            		new JSONObject().put("code", 400).put("msg", "vector length is zero"));
    		 }
    		 if (id==null || id.length() == 0) {
    			 throw new IndexServiceException(
            		new JSONObject().put("code", 400).put("msg", "id is null or has zero length"));
    		 }
    	} catch (JSONException e) {
			throw new IndexServiceException(e);
    	}
    	
        if (threshold == 0)
        	threshold = visualDistanceThreshold;

        double[] vector = bytesToDouble(bvector);

        AbstractSearchStructure linearIndex = linearIndices.get(collection);
        AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
        
        // first query
        Answer answer = null;
        try {
        	answer = ivfpqIndex.computeNearestNeighbors(numResults * pageNum, vector);
        } catch (Exception e) {
            try {
				throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", e.getMessage()));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }

        // then index
        boolean success = index(id, vector, linearIndex, ivfpqIndex);
        if (!success) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "cannot index vector"));
			} catch (JSONException e) {
				throw new IndexServiceException(e);
			}
        }

        String simResSet = getResults(answer, pageNum, numResults, threshold);
        return simResSet;
    }
    
    @GET
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/query_url/{collection}")
    public String queryURL(@QueryParam("url") String url,
    		@PathParam("collection") String collection,
            @DefaultValue("20") @QueryParam("numResults") int numResults,
            @DefaultValue("1") @QueryParam("page") int pageNum,
            @DefaultValue("0") @QueryParam("threshold") double threshold) {
    	
    	JSONObject response = new JSONObject();
    	try {
    			
    		
        	response.put("msg", "Inside method queryURL");
        	
    		if(collection==null || !linearIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
   					new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
    		}
    		
    		response.put("msg1", "Collection Exists!");
    		
    		AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
    		
    		double[] vector = null;
    		try {
    			
    			response.put("url", url);
    			BufferedImage image = null;
    			try {
    				image = fetch(url);
    				//image = fetchWithoutProxy(url);
    			}
    			catch(Exception e) {
    				response.put("exception", e.getMessage());
    			}
    			
    			if(image == null) {
    				response.put("image", "null");
    				throw new IndexServiceException(
							new JSONObject().put("code", 400).put("msg", "image is null"));
    			}
    			else {
    				response.put("size", image.getHeight()+"x"+image.getWidth());
    			}
    			
    			response.put("msg2", "Fetch Done!");
    			
    			vector = extract(image);
    			if(vector == null) {
	    			throw new IndexServiceException(
							new JSONObject().put("code", 400).put("msg", "vector is null"));
	    		}
    			
    			response.put("msg3", "Feature Extraction Done!");
    		}
    		catch(Exception e) {
    			throw new IndexServiceException(
						new JSONObject().put("code", 400).put("exception", e.getMessage()).put("msg", "Error on fetching or frature extraction"));
    		}

    		// first query
            Answer answer = null;
            try {
            	answer = ivfpqIndex.computeNearestNeighbors(numResults * pageNum, vector);
            } catch (Exception e) {
                try {
    				throw new IndexServiceException(
    						new JSONObject().put("code", 400).put("msg", e.getMessage()));
    			} catch (JSONException ex) {
    				throw new IndexServiceException(ex);
    			}
            }

            response.put("msg4", "Compuration of Nearest Neighbors Done!");
            
            String simResSet = getResults(answer, pageNum, numResults, threshold);
            
            response.put("msg5", "Parsing of response Done Done!");
            
            return simResSet;
		}
    	catch (Exception e) {
    		e.printStackTrace();
    	}
    	
    	//return response.toString();
    	JsonResultSet emptyResult = new JsonResultSet();
    	return emptyResult.toJSON();
    }
    
    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/qindex_url/{collection}")
    public String qIndexURL(
    		@FormDataParam("url") String url,
    		@FormDataParam("id") String id,
    		@PathParam("collection") String collection,
            @DefaultValue("20") @FormDataParam("numResults") int numResults,
            @DefaultValue("1") @FormDataParam("page") int pageNum,
            @DefaultValue("0") @FormDataParam("threshold") double threshold) {
    	
    	try {
    		if(collection==null || !linearIndices.containsKey(collection) || !ivfpqIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
   					new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
    		}
    		if (id==null || id.length() == 0) {
    			throw new IndexServiceException(
    					new JSONObject().put("code", 400).put("msg", "id is null or has zero length"));
   		 	}
    		
    		JSONObject response = new JSONObject();
    		
    		AbstractSearchStructure linearIndex = linearIndices.get(collection);
    		AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
    		
    		double[] vector = null;
    		try {
    			BufferedImage image = fetch(url);
    			vector = extract(image);
    			if (vector==null || vector.length == 0) {
    				response.put("e1", "vector length is zero");
    	   		}
    		}
    		catch(Exception e) {
    			response.put("e2", e.getMessage());
    		}
    		
    		// first query
            Answer answer = null;
            try {
            	answer = ivfpqIndex.computeNearestNeighbors(numResults * pageNum, vector);
            } catch (Exception e) {
            	response.put("e3", e.getMessage());
            	response.put("e4", "Cannot compute NN!");
            }

         // then index
            try {
            	boolean success = index(id, vector, linearIndex, ivfpqIndex);
            	if (!success) {
            		response.put("e4", "Index failes!");
            	}
            } catch (JSONException e) {
            	response.put("e5", e.getMessage());
			}

            String simResSet = getResults(answer, pageNum, numResults, threshold);
            return simResSet;
            
		}
    	catch (Exception e) {
			throw new IndexServiceException(e);
		}
    }
    
    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/query_vector/{collection}")
    public String queryVector(@FormDataParam("vector") byte[] bvector,
    		@PathParam("collection") String collection,
            @DefaultValue("20") @FormDataParam("numResults") int numResults,
            @DefaultValue("1") @FormDataParam("page") int pageNum,
            @DefaultValue("0") @FormDataParam("threshold") double threshold) {
        // validate parameters
    	try {
    		if(collection==null || !linearIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
   		 	}	
   		 	if (pageNum <= 0) {
				throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "page number < 0"));
   		 	}
   		 	if (pageNum * numResults > 1000) {
   		 		throw new IndexServiceException(
   		 				new JSONObject().put("code", 400).put("msg", "pageNum * numResults > 1000"));
   		 	}
   		 	if (bvector.length == 0) {
   		 		throw new IndexServiceException(
   		 				new JSONObject().put("code", 400).put("msg", "vector length is zero"));
   		 	}
    	} catch (JSONException e) {
			throw new IndexServiceException(e);
    	}
        
    	if (threshold == 0) 
        	threshold = visualDistanceThreshold;
        
        
        AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
        
        double[] vector = bytesToDouble(bvector);

        // first query
        Answer answer = null;
        try {
        	answer = ivfpqIndex.computeNearestNeighbors(numResults * pageNum, vector);
        } catch (Exception e) {
            e.printStackTrace();
            try {
				throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", e.getMessage()));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }

        String results = getResults(answer, pageNum, numResults, threshold);
        return results;
    }

    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/query_id/{collection}")
    public String queryId(
    		@FormDataParam("id") String id,
    		@PathParam("collection") String collection,
            @DefaultValue("20") @FormDataParam("numResults") int numResults,
            @DefaultValue("1") @FormDataParam("page") int pageNum,
            @DefaultValue("0") @FormDataParam("threshold") double threshold) {
        // validate parameters
    	try {
    		if(collection==null || !linearIndices.containsKey(collection) || !ivfpqIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
   		 	}	
   		 	if (pageNum <= 0) {
				throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "page number < 0"));
   		 	}
   		 	if (pageNum * numResults > 1000) {
   		 		throw new IndexServiceException(
   		 				new JSONObject().put("code", 400).put("msg", "pageNum * numResults > 1000"));
   		 	}
   		 	if (id==null || id.length() == 0) {
   		 		throw new IndexServiceException(
   		 				new JSONObject().put("code", 400).put("msg", "id is null or has zero length"));
   		 	}
    	} catch (JSONException e) {
    		throw new IndexServiceException(e);
    	}
    	
        if (threshold == 0) {
        	threshold = visualDistanceThreshold;
        }

        AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
        Linear linearIndex = (Linear) linearIndices.get(collection);
        
        // first query
        Answer answer = null;
        try {
        	double[] vector = null;
        	int iid = linearIndex.getInternalId(id);
			if(iid>0)
        		vector  = linearIndex.getVector(iid);
        	if(vector != null){
        		answer = ivfpqIndex.computeNearestNeighbors(numResults * pageNum, vector);
        	}
        	else {
        		throw new Exception("Vector " + id + " does not exist");
        	}
        } catch (Exception e) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", e.getMessage()));
			} catch (JSONException e1) {
				throw new IndexServiceException(e);
			}
        }

        String results = getResults(answer, pageNum, numResults, threshold);
        return results;
    }


    
    
    
    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.TEXT_PLAIN)
    @Path("/index/{collection}")
    public String indexVector(
    		@FormDataParam("id") String id,
    		@PathParam("collection") String collection,
            @FormDataParam("vector") byte[] bvector) {

    	try {
    		if(collection==null || !linearIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
   		 	}	
   		 	if (id==null || id.length() == 0) {
   		 		throw new IndexServiceException(
   		 				new JSONObject().put("code", 400).put("msg", "id is null or has zero length"));
   		 	}
   		 if (bvector.length == 0) {
		 		throw new IndexServiceException(
		 				new JSONObject().put("code", 400).put("msg", "vector length is zero"));
		 	}
    	} catch (JSONException e) {
    		throw new IndexServiceException(e);
    	}

    	AbstractSearchStructure linearIndex = linearIndices.get(collection);
    	AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
    	
        double[] vector = bytesToDouble(bvector);

        boolean success = index(id, vector, linearIndex, ivfpqIndex);
        if (!success) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", "cannot index"));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }

        return "{\"success\" : " + true + "}";
    }
    

    @GET
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/vector/{collection}")
    public String getVector(@QueryParam("id") String id,
    		@PathParam("collection") String collection) {

    	try {
    		if(collection==null || !linearIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
						new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
   		 	}	
   		 	if (id==null || id.length() == 0) {
   		 		throw new IndexServiceException(
   		 				new JSONObject().put("code", 400).put("msg", "id is null or has zero length"));
   		 	}
   		 	
   	    	Linear linearIndex = (Linear) linearIndices.get(collection);
   	   
   	    	JSONArray array = new JSONArray();
   		 	
   	    	int iid = linearIndex.getInternalId(id);
   	    	if(iid >= 0) {
   	    		double[] vector = linearIndex.getVector(iid);
   	    		if(vector != null) {
   	    			
   	    			try {
   	    				JSONObject obj = new JSONObject();
   	    				
   	    				for(double v : vector) {
   	    					array.put(v);
   	    				}
   	    				obj.put("vector", array);
   	    				obj.put("exists", true);
   	    				
   	    				return array.toString();
   	    				//return obj.toString();
   	    			}
   	    			catch(Exception e) {
   	    				
   	    			}
   	    		}
   	    	}
   		 	
   	    	return array.toString();
   	    	
    	} catch (Exception e) {
    		throw new IndexServiceException(e);
    	}

    	
    	//return "{\"exists\" : " + false + "}";
        
    	
    }
    
    
    /**
     * Get information and statistics of the service
     * @return
     * @throws JSONException 
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/statistics/")
    public String statistics() {
    		
        try {    
        	
        	JSONObject response = new JSONObject();
        	
        	for(Entry<String, AbstractSearchStructure> e : linearIndices.entrySet()) {
        		
        		String collection = e.getKey();
            	AbstractSearchStructure linearIndex = linearIndices.get(collection);
            	
            	AbstractSearchStructure ivfpqIndex = e.getValue();
                int ivfpqIndexCount = ivfpqIndex.getLoadCounter();
                int linearIndexCount = linearIndex.getLoadCounter();

                response.put(collection+"_ivfpqIndex", ivfpqIndexCount);
                response.put(collection+"_linearIndex", linearIndexCount);
        	}
        	
            
            @SuppressWarnings("unchecked")
			Enumeration<String> parameters = _context.getInitParameterNames();
            while(parameters.hasMoreElements()) { 
            	String paramKey = parameters.nextElement();
            	String paramValue = _context.getInitParameter(paramKey);
            	response.put(paramKey, paramValue);
            }
            
            return response.toString();
        } catch (Exception e) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", e.getMessage()));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }
    }
    
    /**
     * Get information and statistics of the service
     * @return
     * @throws JSONException 
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/statistics/{collection}")
    public String statistics(
    		@PathParam("collection") String collection) {
    		
        try {    
        	if(collection==null || !linearIndices.containsKey(collection)) {
   			 throw new IndexServiceException(
   					new JSONObject().put("code", 400).put("msg", "collection " + collection + " does not exists"));
        	}
        	
        	JSONObject response = new JSONObject();
        	
        	AbstractSearchStructure ivfpqIndex = ivfpqIndices.get(collection);
        	AbstractSearchStructure linearIndex = linearIndices.get(collection);
        	
            int ivfpqIndexCount = ivfpqIndex.getLoadCounter();
            int linearIndexCount = linearIndex.getLoadCounter();

            response.put("ivfpqIndex", ivfpqIndexCount);
            response.put("linearIndex", linearIndexCount);
            
            @SuppressWarnings("unchecked")
			Enumeration<String> parameters = _context.getInitParameterNames();
            while(parameters.hasMoreElements()) { 
            	String paramKey = parameters.nextElement();
            	String paramValue = _context.getInitParameter(paramKey);
            	response.put(paramKey, paramValue);
            }
            
            return response.toString();
        } catch (Exception e) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", e.getMessage()));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }
    }
    
    /**
     * Get information and statistics of the service
     * @return
     * @throws JSONException 
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/add/{collection}")
    public String addCollection(@PathParam("collection") String collection) {
    		
    	String ivfpqIndexFolder = dataFolder + "/" + collection + "/ivfpq";
		String linearIndexFolder = dataFolder + "/" + collection + "/linear";
		
		String msg = "";
		
        try {    
        	if(linearIndices.containsKey(collection) || ivfpqIndices.containsKey(collection)) {
   			 	throw new IndexServiceException(
   					new JSONObject().put("code", 400).put("msg", "collection " + collection + " already exists"));
        	}	
    			
    		try {
    			File jeLck = new File(ivfpqIndexFolder, "je.lck");
    			if(jeLck.exists()) {
    				jeLck.delete();
    			}
    			jeLck = new File(linearIndexFolder, "je.lck");
    			if(jeLck.exists()) {
    				jeLck.delete();
    			}
    			
    			msg += " je.lck deleted!  Create linear to: " + linearIndexFolder;
    			
    			Linear linearIndex = new Linear(targetLengthMax, numOfVectors, false, linearIndexFolder, false, true, 0);
    				
    			msg += " Linear index created!  ";
    			
    			IVFPQ ivfpqIndex = new IVFPQ(targetLength, numOfVectors, false, ivfpqIndexFolder, subvectorsNum, kc,
    						PQ.TransformationType.RandomPermutation, numCoarseCentroids, true, 0);
    			
    			msg += " IVFPQ index created!  ";
    			
    			ivfpqIndex.loadCoarseQuantizer(coarseQuantizerFile);
    			msg += " Coarse Quantizer loaded!  ";
    			
    			ivfpqIndex.loadProductQuantizer(productQuantizerFile);
    			msg += " Product Quantizer loaded!  ";
    			
    			// how many (out of 8192) lists should be visited during search.
    			ivfpqIndex.setW(w); 
    				
    			if(linearIndices != null && ivfpqIndices != null) {
    				linearIndices.put(collection, linearIndex);
    				ivfpqIndices.put(collection, ivfpqIndex);
    			}
    		}
    		catch(Exception e) {
        		throw new IndexServiceException(
        				new JSONObject().put("code", 501).put("msg", "error on startup: " + e.getMessage()));
    		}
            return "{ \"" + collection + "\" :  \"created\" }";
        } catch (Exception e) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", msg));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }
    }
    
    /**
     * Get information and statistics of the service
     * @return
     * @throws JSONException 
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/delete/{collection}")
    public String deleteCollection(@PathParam("collection") String collection) {
    		
    	String collectionFolder = dataFolder + "/" + collection;
		
        try {    
    			
    		try {
    			AbstractSearchStructure index = linearIndices.remove(collection);
    			if(index  != null)
    				index.close();
    		}
    		catch(Exception e) { }
    		try {
    			AbstractSearchStructure index = ivfpqIndices.remove(collection);
    			if(index  != null)
    				index.close();
    		}
    		catch(Exception e) { }
    		
    		File dir = new File(collectionFolder);
			if(dir.exists()) {
				FileUtils.deleteDirectory(dir);
			}
			
            return "{ \"" + collection + "\" :  \"deleted\" }";
        } catch (Exception e) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", "Cannot delete "+collection));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }
    }
    
    /**
     * Get information and statistics of the service
     * @return
     * @throws JSONException 
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/remove/{collection}")
    public String removeCollection(@PathParam("collection") String collection) {
    		
        try {    
    			
    		try {
    			AbstractSearchStructure index = linearIndices.remove(collection);
    			if(index  != null) {
    				index.close();
    			}
    		}
    		catch(Exception e) { }
    		try {
    			AbstractSearchStructure index = ivfpqIndices.remove(collection);
    			if(index  != null) {
    				index.close();
    			}
    		}
    		catch(Exception e) { }

			
            return "{ \"" + collection + "\" :  \"removed\" }";
        } catch (Exception e) {
        	try {
				throw new IndexServiceException(
						new JSONObject().put("code", 405).put("msg", "Cannot remove " + collection));
			} catch (JSONException ex) {
				throw new IndexServiceException(ex);
			}
        }
    }
    
    private boolean index(String id, double[] vector, AbstractSearchStructure linearIndex, AbstractSearchStructure ivfpqIndex) {
    	boolean success = true;
    	try {
            success = success && linearIndex.indexVector(id, vector);
            
            double[] newVector = Arrays.copyOf(vector, targetLength);
			if (newVector.length < vector.length) {
				Normalization.normalizeL2(newVector);
			}
            success = success && ivfpqIndex.indexVector(id, newVector);
        } catch (Exception e) {
        	success = false;
        }
    	return success;
    }
    
    /**
     * Transform a ByteArray to Doubles
     * @param bytes
     * @return
     */
    private static double[] bytesToDouble(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        double[] doubles = new double[bytes.length / 8];
        for (int i = 0; i < doubles.length; i++)
            doubles[i] = buffer.getDouble(i * 8);
        return doubles;
    }
    
    private String getResults(Answer answer, int pageNum, int numResults, double threshold) {
    	Result[] res = answer.getResults();
    	  
    	JsonResultSet resultsresp = new JsonResultSet();
        
        int startIndex = Math.min(numResults * (pageNum - 1), res.length);
        int endIndex = Math.min(numResults * pageNum, res.length);
        
        for (int i = startIndex; i < endIndex; i++) {
            if (res[i].getDistance() > threshold)
                break;
            resultsresp.addResult(res[i].getId(), (i + 1), res[i].getDistance());
        }
        
        return resultsresp.toJSON();
    }
    
    /**
     * Fetch Media Content 
     * @param url
     * @return BufferedImage
     * @throws ClientProtocolException 
     * @throws IOException
     * @throws JSONException 
     */
    private BufferedImage fetch(String url) throws ClientProtocolException, IOException, JSONException { 	
    	
    	//URL urlObject = new URL(url);
    	HttpGet httpget = new HttpGet(url);
    	 
    	//HttpHost target = new HttpHost(urlObject.getAuthority(), urlObject.getPort(), urlObject.getProtocol());
    	//HttpGet request = new HttpGet(urlObject.getPath());
    	
    	//json.put("target", target.toHostString());
    	//Object proxy = httpClient.getParams().getParameter(ConnRoutePNames.DEFAULT_PROXY);
    	
    	//json.put("request", ("Executing request " + request.getRequestLine() + " to " + target + " via " + proxy));
    	
    	HttpResponse response = null;
    	synchronized(httpClient) {
    		//response = httpClient.execute(target, request);
    		response = httpClient.execute(httpget);	
    	}
    	
    	if(response != null) { 
    		//json.put("response", response.getStatusLine());
    		
    		HttpEntity entity = response.getEntity();
    		if (entity != null) {

    			InputStream contentStream = entity.getContent();
    			byte[] content = IOUtils.toByteArray(contentStream);
    			 
    			//json.put("content", content.length);
    			
    			ByteArrayInputStream is = new ByteArrayInputStream(content);
    			BufferedImage image = ImageIO.read(is);
    			 
    			return image;
    		}
    		else {
    			//json.put("entity", "null");
    		}
    	}
    	else {
    		//json.put("response", "null");
    	}
    	
    	return null;
    }
    
    /**
     * Extract feature vector of an image
     * @param image
     * @return double[]
     * @throws Exception
     */
    private double[] extract(BufferedImage image) throws Exception {

    	ImageVectorization imvec = new ImageVectorization(null, image, targetLengthMax, maxNumPixels);
    		
		ImageVectorizationResult imvr = imvec.call();
		double[] vector = imvr.getImageVector();

    	return vector;
    }
    
    
}
