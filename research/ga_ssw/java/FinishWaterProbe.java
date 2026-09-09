// Staged interface reproduction, NOT a bitwise restart of the original main.
// Reconstructs archive from completed physical runs; original jars do all search/classification.
import app_ssw_ga.*;
import configure.ConfigureRead;
import contruction.*;
import nna.Classify;
import other.ArcFile;
import java.nio.file.*;
import java.util.*;

public class FinishWaterProbe {
    static List<Model> clean(List<Model> models) {
        Collections.sort(models);
        SSWGaSupport.handleClusterTem(models); TemH2O.handleH2OCluster(models);
        SSWGaSupport.checkMonComplete(models,3.0); return models;
    }
    static List<ModelSimS> classify(List<Model> models,List<ConInfo> refs) throws Exception {
        return SSWGaSupport.filterHighEnergy(Classify.removeDuplicate(
            SSWGaSupport.getSims(clean(models),refs),ConfigureRead.getSim()),ConfigureRead.getMaxSamplingEnergy());
    }
    static void requireComplete(Path dir) throws Exception {
        if(!Files.readString(dir.resolve("lasp.out")).contains("SSW all done") ||
            ArcFile.getAllModelByArcLowMemory(dir.resolve("all.arc").toString()).isEmpty())
            throw new IllegalStateException("Incomplete LASP task: "+dir);
    }
    public static void main(String[] args) throws Exception {
        ConfigureRead.load();
        if(ConfigureRead.getType()!=3 || ConfigureRead.getPopClassifyNum()!=1)
            throw new IllegalArgumentException("Probe supports supplied water/k=1 only");
        if(ConfigureRead.getCombineMultiUnitNum()!=1 || ConfigureRead.getMinGA()!=6 ||
           ConfigureRead.getQuickSSWIterations()!=1 || ConfigureRead.getFineSSWIterations()!=1 ||
           ConfigureRead.getQuickSSWStep()!=2 || ConfigureRead.getFineSSWStep()!=2 ||
           ConfigureRead.getOPTSSWStep()!=1 || ConfigureRead.getCPU()!=1 ||
           ConfigureRead.getTaskNum()!=1 || ConfigureRead.getSSWTaskNum()!=1)
            throw new IllegalArgumentException("Requires recorded one-initial/six-offspring smoke configuration");
        if(ConfigureRead.getComponent().size()!=2 ||
           !Integer.valueOf(30).equals(ConfigureRead.getComponent().get("H")) ||
           !Integer.valueOf(15).equals(ConfigureRead.getComponent().get("O")) || ConfigureRead.getIfPer()!=0)
            throw new IllegalArgumentException("Requires nonperiodic (H2O)15 composition");
        Path root=Path.of(ConfigureRead.getRootPath());
        requireComplete(root.resolve("output/initial/0"));
        if(Files.exists(root.resolve("output/initial/1")) || Files.exists(root.resolve("output/iterate/0/opt/6")))
            throw new IllegalArgumentException("Unexpected extra upstream tasks");
        List<ConInfo> refs=ConfigureRead.getBaseCon();
        List<ModelSimS> archive=new ArrayList<>(classify(ArcFile.getAllModelByArcLowMemory(
            root.resolve("output/initial/0/all.arc").toString()),refs));
        List<Model> offspring=new ArrayList<>();
        for(int i=0;i<6;i++) {
            Path dir=root.resolve("output/iterate/0/opt/"+i);
            requireComplete(dir);
            List<Model> rows=ArcFile.getAllModelByArcLowMemory(dir.resolve("all.arc").toString());
            Collections.sort(rows);offspring.add(rows.get(0));
        }
        Classify.mergeModelSimsBySimilarity(archive,classify(offspring,refs),ConfigureRead.getSim());
        System.out.println("Restored archive size: "+archive.size());
        for(String stage:new String[]{"staged-quick","staged-fine"}) {
            Path output=root.resolve("output/"+stage);
            if(Files.exists(output))throw new IllegalStateException("Refusing overwrite: "+output);
            List<List<ModelSimS>> regions=SSWGaSupport.selectParentsByKMeans(archive,1);
            List<Model> seeds=new ArrayList<>();seeds.add(regions.get(0).get(0).getModel());
            // k=1: score ranking cannot change region allocation.
            int steps=stage.equals("staged-quick")?ConfigureRead.getQuickSSWStep():ConfigureRead.getFineSSWStep();
            List<Model> found=SSWGaSupport.SSWExplore(seeds,output.toString(),steps,ConfigureRead.getSSWTemper());
            requireComplete(output.resolve("0"));
            Classify.mergeModelSimsBySimilarity(archive,classify(found,refs),ConfigureRead.getSim());
            Collections.sort(archive);
            System.out.println(stage+" completed; archive="+archive.size()+" best="+archive.get(0).getModel().getEnergy());
        }
        SSWGaSupport.finalHandle(archive);
        System.out.println("Staged interface workflow completed");
    }
}
