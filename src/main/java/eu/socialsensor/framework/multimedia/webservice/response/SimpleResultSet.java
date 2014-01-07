package eu.socialsensor.framework.multimedia.webservice.response;

import java.util.ArrayList;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlList;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name = "queryResponse")
public class SimpleResultSet {

    @XmlAttribute
    private int numResults;


    //@XmlElement(name = "results")
    @XmlList
    private ArrayList<SimpleResult> results;

    public SimpleResultSet() {
    	
    }

    public SimpleResultSet(ArrayList<SimpleResult> results) {
        this.results = results;
        this.numResults = results.size();
    }

    public void setResults(ArrayList<SimpleResult> results) {
        this.results = results;
    }
}
