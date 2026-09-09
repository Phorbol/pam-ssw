// Original SGN classifier only: avoids unrelated degenerate equal-energy weights.
import contruction.*;
import nna.Classify;
import java.util.*;
import java.nio.file.*;

public class ProjectionProbe {
    public static void main(String[] args) throws Exception {
        List<ModelSimS> rows=new ArrayList<>();
        for(String line:Files.readAllLines(Path.of(args[0]))) {
            String[] p=line.split("\\s+"); Model m=new Model(); m.setEnergy(Double.parseDouble(p[0]));
            List<Double> sims=new ArrayList<>(); for(int i=1;i<p.length;i++)sims.add(Double.parseDouble(p[i]));
            ModelSimS row=new ModelSimS(); row.setModel(m); row.setSims(sims); rows.add(row);
        }
        double tol=Double.parseDouble(args[2]);
        String out="{\"original_reverse_same\":"+Classify.isSim(rows.get(0),rows.get(1),tol)
            +",\"original_rigid_same\":"+Classify.isSim(rows.get(0),rows.get(2),tol)
            +",\"dedup_count\":"+Classify.removeDuplicate(rows,tol).size()+"}\n";
        Files.writeString(Path.of(args[1]),out);
    }
}
