// Deterministic oracle against uploaded sgn.jar, not recompiled decompilation.
import contruction.AtoCoo;
import contruction.Model;
import ga_molecular_crystal.MC_Base;
import ga_monomer.MonomerBase;
import contruction.AtomUtil;
import java.util.*;
import java.nio.file.*;
public class Type3DockingProbe {
  public static void main(String[] args) throws Exception {
    List<String> lines=Files.readAllLines(Path.of(args[0]));
    List<Model> models=new ArrayList<>();
    for(int part=0;part<2;part++) {
      List<AtoCoo> atoms=new ArrayList<>();
      for(int i=part*3;i<part*3+3;i++) {
        String[] r=lines.get(i).trim().split("\\s+");
        AtoCoo a=new AtoCoo();a.setAtomic(Integer.parseInt(r[0]));
        a.setCoordinate(new double[]{Double.parseDouble(r[1]),Double.parseDouble(r[2]),Double.parseDouble(r[3])});atoms.add(a);
      }
      Model m=new Model();m.setAtoCoos(atoms);
      m.setMonCompose(new ArrayList<>(List.of(new ArrayList<>(List.of(0,1,2)))));models.add(m);
    }
    Model out;
    if(args.length>1 && args[1].equals("combine")) {
      Class<?> holder=Class.forName("java.lang.Math$RandomNumberGeneratorHolder");
      var field=holder.getDeclaredField("randomNumberGenerator");field.setAccessible(true);
      ((Random)field.get(null)).setSeed(555);
      List<AtomUtil> units=new ArrayList<>();
      for(Model model:models) {
        double[] mean=new double[3];
        for(AtoCoo a:model.getAtoCoos())for(int j=0;j<3;j++)mean[j]+=a.getCoordinate()[j]/3.;
        for(AtoCoo a:model.getAtoCoos())for(int j=0;j<3;j++)a.getCoordinate()[j]-=mean[j];
        units.add(MonomerBase.toAtomUtil(model.getAtoCoos()));
      }
      out=MonomerBase.fitMonomerCombination(1.5,units,10);
      Random replay=new Random(555);
      for(int i=0;i<40;i++)System.out.printf(Locale.ROOT,"DRAW %.17g%n",replay.nextDouble());
    } else out=MC_Base.fitBinding(1.5,models.get(0),models.get(1),5);
    for(AtoCoo a:out.getAtoCoos()) {
      double[] x=a.getCoordinate();
      System.out.printf(Locale.ROOT,"ATOM %d %.17g %.17g %.17g%n",a.getAtomic(),x[0],x[1],x[2]);
    }
  }
}
