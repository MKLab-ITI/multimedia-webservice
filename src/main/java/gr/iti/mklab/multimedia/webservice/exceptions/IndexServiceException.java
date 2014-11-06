package gr.iti.mklab.multimedia.webservice.exceptions;

import javax.ws.rs.WebApplicationException;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.codehaus.jettison.json.JSONObject;

public class IndexServiceException extends WebApplicationException {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public IndexServiceException(JSONObject jsonObject) {
	    super(Response.status(Response.Status.OK)
	            .entity(jsonObject)
	            .type(MediaType.APPLICATION_JSON)
	            .build());
	}

	public IndexServiceException(Exception e) {
	    super(e);
	}
	
}
