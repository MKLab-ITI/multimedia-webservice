package gr.iti.mklab.multimedia.webservice;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

import javax.servlet.ServletContext;
import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.WebApplicationException;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response.ResponseBuilder;

import com.sun.jersey.core.spi.factory.ResponseBuilderImpl;
import com.sun.jersey.multipart.FormDataParam;

/**
 * This class is an image upload web-service. Accepts an image with post and returns the url of the
 * saved image.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * 
 */
// Sets the path to base URL + /
@Path("images/")
public class UploadController {

	private @Context ServletContext context;
	
    private static String uploadFolder;
    private static String uploadBrowseFolder;
    
    public UploadController(@Context ServletContext context) throws Exception {
    	
    	this.context = context;
    	uploadFolder = context.getInitParameter(uploadFolder);
      
        createFolder(uploadFolder);
    }

    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/upload")
    public String saveImage(@FormDataParam("photo") File imageFile, @FormDataParam("id") String id) {
        if (imageFile.length() == 0) {
        	ResponseBuilder response = new ResponseBuilderImpl();
        	Exception ex = new Exception("image is of zero length!");
            response.status(401);
            response.entity(ex.getStackTrace());
            throw new WebApplicationException(ex, response.build());
        }
        if (id.length() == 0) {
        	Exception ex = new Exception("id is of zero length!");
        	ResponseBuilder response = new ResponseBuilderImpl();
            response.status(402);
            response.entity(ex.getStackTrace());
            throw new WebApplicationException(ex, response.build());
        }

        try {
            copyFile(imageFile, new File(uploadFolder, id));
        } catch (IllegalArgumentException e) {
        	ResponseBuilder response = new ResponseBuilderImpl();
            response.status(403);
            response.entity(e.getStackTrace());
            throw new WebApplicationException(e, response.build()); 
        } catch (Exception e) {
        	ResponseBuilder response = new ResponseBuilderImpl();
            response.status(400);
            response.entity(e.getStackTrace());
            throw new WebApplicationException(e, response.build());
        }
        return uploadBrowseFolder + id;
    }
    
    private static void copyFile(File sourceFile, File destFile) throws Exception {
        if (!destFile.exists()) {
            destFile.createNewFile();
        } else {
            throw new IllegalArgumentException("A file with this id already exists!");
        }

        FileInputStream source = null;
        FileOutputStream destination = null;

        try {
            source = new FileInputStream(sourceFile);
            destination = new FileOutputStream(destFile);
            destination.getChannel().transferFrom(source.getChannel(), 0, source.getChannel().size());
        } 
        catch(Exception e) {
        	
        }
        finally {
            if (source != null) {
                source.close();
            }
            if (destination != null) {
                destination.close();
            }
        }
    }

    private static void createFolder(String uploadsFolder) {
        // check if the upload folder exists and if not create it
        File uploadDir = new File(uploadsFolder);
        if (!uploadDir.exists()) {
            System.out.println("Creating directory: " + uploadsFolder);
            boolean result = uploadDir.mkdir();
            if (result) {
                System.out.println("DIR created");
            }
        }
    }
}