// Calls the supplied SGN bytecode on real-structure feature/energy records.
import app_ssw_ga.SSWGaSupport;
import contruction.*;
import nna.Classify;
import ga_cluster_cell.Compete;
import java.util.*;
import java.nio.file.*;

public class CoreProbe {
    static class Weights extends Compete {
        Weights(List<Model> m){super(m);}
        double[] cumulative(){return pe_fib;}
    }
    static List<Integer> ids(List<ModelSimS> rows,IdentityHashMap<ModelSimS,Integer> index){
        List<Integer> out=new ArrayList<>();for(ModelSimS m:rows)out.add(index.get(m));return out;
    }
    public static void main(String[] args) throws Exception {
        List<ModelSimS> rows=new ArrayList<>();IdentityHashMap<ModelSimS,Integer> index=new IdentityHashMap<>();
        for(String line:Files.readAllLines(Path.of(args[0]))) {
            String[] p=line.split("\\s+");Model m=new Model();m.setEnergy(Double.parseDouble(p[0]));
            List<Double> s=new ArrayList<>();for(int i=1;i<p.length;i++)s.add(Double.parseDouble(p[i]));
            ModelSimS r=new ModelSimS();r.setModel(m);r.setSims(s);index.put(r,rows.size());rows.add(r);
        }
        double tolerance=Double.parseDouble(args[2]),window=Double.parseDouble(args[3]);
        List<List<Boolean>> same=new ArrayList<>();for(ModelSimS a:rows){List<Boolean> b=new ArrayList<>();for(ModelSimS c:rows)b.add(Classify.isSim(a,c,tolerance));same.add(b);}
        List<ModelSimS> dedup=Classify.removeDuplicate(rows,tolerance);
        List<ModelSimS> merged=new ArrayList<>(rows.subList(0,2));Classify.mergeModelSimsBySimilarity(merged,rows.subList(2,rows.size()),tolerance);
        List<List<Integer>> clusters=new ArrayList<>();for(List<ModelSimS> group:SSWGaSupport.selectParentsByKMeans(rows,1))clusters.add(ids(group,index));
        List<Model> models=new ArrayList<>();for(ModelSimS x:rows)models.add(x.getModel());
        String out="{\"same\":"+same+",\"dedup\":"+ids(dedup,index)+",\"merged\":"+ids(merged,index)+",\"window\":"+ids(SSWGaSupport.filterHighEnergy(rows,window),index)+",\"one_cluster\":"+clusters+",\"compete_cumulative\":"+Arrays.toString(new Weights(models).cumulative())+"}\n";
        Files.writeString(Path.of(args[1]),out);
    }
}
