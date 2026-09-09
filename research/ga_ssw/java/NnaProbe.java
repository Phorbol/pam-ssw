// Reference fixture exporter: invokes original NNA bytecode, not recompiled decompilation.
import contruction.*;
import nna.*;
import nnaPar.NeiSimP;
import other.ArcFile;
import java.util.*;
import java.nio.file.*;

public class NnaProbe {
    static String json(Object x) {
        if(x==null) return "null";
        if(x instanceof String) return "\""+((String)x).replace("\\","\\\\").replace("\"","\\\"")+"\"";
        if(x instanceof Number || x instanceof Boolean) return x.toString();
        if(x instanceof Map) {
            List<String> a=new ArrayList<>();
            for(Object o:((Map<?,?>)x).entrySet()) {Map.Entry<?,?> e=(Map.Entry<?,?>)o;a.add(json(e.getKey())+":"+json(e.getValue()));}
            return "{"+String.join(",",a)+"}";
        }
        List<String> a=new ArrayList<>();
        if(x instanceof Iterable) for(Object v:(Iterable<?>)x)a.add(json(v));
        else if(x.getClass().isArray()) for(int i=0;i<java.lang.reflect.Array.getLength(x);i++)a.add(json(java.lang.reflect.Array.get(x,i)));
        else throw new IllegalArgumentException(x.getClass().toString());
        return "["+String.join(",",a)+"]";
    }
    static Map<String,Object> descriptor(ConInfo c) {
        Map<String,Object> d=new LinkedHashMap<>();
        d.put("n1",c.getConfigureInfo().getAllCns1());d.put("n2",c.getConfigureInfoSec().getAllCns2());
        d.put("n3",c.getConfigureInfoSec().getAllCns3());d.put("d1",c.getConfigureInfo().getAllBD1());
        d.put("d2",c.getConfigureInfoSec().getBaseD2());d.put("d3",c.getConfigureInfoSec().getBaseD3());return d;
    }
    static Model transform(Model m,int mode) {
        Model n=new Model(); n.setEnergy(m.getEnergy());n.setPbc(m.getPbc());
        List<AtoCoo> atoms=new ArrayList<>();
        for(AtoCoo a:m.getAtoCoos()) {
            AtoCoo b=new AtoCoo();b.setAtomic(a.getAtomic());double[] p=a.getCoordinate();
            b.setCoordinate(mode==2?new double[]{-p[1]+2.0,p[0]-3.0,p[2]+1.0}:p.clone());atoms.add(b);
        }
        if(mode==1)Collections.reverse(atoms);
        n.setAtoCoos(atoms);return n;
    }
    public static void main(String[] args) throws Exception {
        String path=args[0];int periodic=Integer.parseInt(args[1]);double range=Double.parseDouble(args[2]);
        int type=Integer.parseInt(args[3]);double grid=Double.parseDouble(args[4]);int limit=Integer.parseInt(args[5]);
        List<Model> originals=ArcFile.getAllModelByArcLowMemory(path);
        List<Model> models=new ArrayList<>(); List<String> labels=new ArrayList<>();
        for(int i=0;i<Math.min(limit,originals.size());i++)for(int mode=0;mode<(periodic==0?3:2);mode++) {
            models.add(transform(originals.get(i),mode));labels.add(i+":"+new String[]{"original","reverse","rotate_translate"}[mode]);
        }
        List<Integer> elements=BasicInfo.getAtoms(models.get(0).getAtoCoos());
        HashMap<int[],Double> bonds=BasicInfo.getPredictedBondLength(models.get(0));
        List<ConInfo> cs=new ArrayList<>();List<Map<String,Object>> frames=new ArrayList<>();
        for(int k=0;k<models.size();k++) {
            Model m=models.get(k);
            ConInfo c=periodic==1?Neighbour.quickGetConInfo(m,elements,bonds,1,type,range,grid,3):Neighbour.getConInfo(m,elements,bonds,0,range,3);
            cs.add(c);Map<String,Object> frame=new LinkedHashMap<>();frame.put("label",labels.get(k));frame.put("energy",m.getEnergy());
            List<Integer> z=new ArrayList<>();List<double[]> xyz=new ArrayList<>();
            for(AtoCoo a:m.getAtoCoos()){z.add(a.getAtomic());xyz.add(a.getCoordinate());}
            frame.put("cell_lengths",m.getPbc().getAbc());frame.put("cell_angles_degrees",m.getPbc().getAbcA());
            frame.put("numbers",z);frame.put("positions",xyz);frame.put("descriptor",descriptor(c));frames.add(frame);
        }
        if(args.length>7 && args[7].equals("batch")) {
            System.out.println("Entering original cpu=2 batch with "+models.size()+" real frames");System.out.flush();
            new NeiSimP(models,2,periodic==1,range,type,grid,3,Arrays.asList(cs.get(0),cs.get(0),cs.get(0)),new double[]{.3,.2,.2,.1,.1,.1}).getAllConInfo();
            System.out.println("Batch completed");return;
        }
        List<List<Double>> sim=new ArrayList<>();
        for(ConInfo a:cs){List<Double> row=new ArrayList<>();for(ConInfo b:cs)row.add(Sim.getSim(a,b,.3,.2,.2,.1,.1,.1));sim.add(row);}
        Map<String,Object> out=new LinkedHashMap<>();out.put("source",path);out.put("periodic",periodic==1);out.put("neighbor_range",range);out.put("elements",elements);
        List<List<Number>> pairs=new ArrayList<>();for(int[] p:bonds.keySet())pairs.add(Arrays.asList(p[0],p[1],bonds.get(p)));
        out.put("bond_lengths",pairs);out.put("frames",frames);out.put("similarity_matrix",sim);
        if(args.length>8) {
            List<ConInfo> refs=other.FileHandler.readConInfo(args[8]);
            List<Map<String,Object>> rd=new ArrayList<>();for(ConInfo ref:refs)rd.add(descriptor(ref));
            List<List<Double>> projections=new ArrayList<>();
            for(ConInfo c:cs){List<Double> row=new ArrayList<>();for(ConInfo ref:refs)row.add(Sim.getSim(c,ref,.3,.2,.2,.1,.1,.1));projections.add(row);}
            out.put("reference_descriptors",rd);out.put("projections",projections);
        }

        Files.writeString(Path.of(args[6]),json(out)+"\n");
    }
}
