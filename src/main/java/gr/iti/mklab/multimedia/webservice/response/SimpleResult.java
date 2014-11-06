package gr.iti.mklab.multimedia.webservice.response;

import java.text.DecimalFormat;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;

@XmlRootElement(name = "image")
@XmlType(propOrder = { "name", "rank", "score" })
public class SimpleResult {

    private static DecimalFormat df = new DecimalFormat("#.###");
    
    public SimpleResult() {
    }

    public SimpleResult(String name, int rank, double distance) {
        this.name = name;
        this.rank = rank;
        
        // transform distance into similarity
        double similarity = (2.0 - Math.sqrt(distance)) / 2.0;
        
        // format the score
        this.score = df.format(similarity);
    }

    @XmlElement
    private String name;
    @XmlElement
    private int rank;
    @XmlElement
    private String score;

    public void setName(String name) {
        this.name = name;
    }

    public void setRank(int rank) {
        this.rank = rank;
    }

    public void setScore(String score) {
        this.score = score;
    }

}
