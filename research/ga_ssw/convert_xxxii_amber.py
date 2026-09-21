"""Exact, source-pinned XXXII custom-AMBER to stock CHARMM representation.

No calculator invocation; this is not a general GAFF parameter converter.
"""
import argparse,collections,hashlib,json
from pathlib import Path

DATA_SHA='b1f1707c2203932f4b7619906908645513f20a54fd53af6bf43c697091a550cf'
INPUT_SHA='9586229ab20578640d065687f310140c0b5adbe113ef1905adddc19700a64192'

def digest(raw):return hashlib.sha256(raw).hexdigest()

def sections(text):
    result={};current=None
    for line in text.splitlines():
        content=line.split('#')[0].strip()
        if not content:continue
        if content[0].isalpha():current=content;result[current]=[]
        elif current:result[current].append(content.split())
    return result

def verify_topology(s):
    coeff={int(r[0]):tuple(map(float,r[1:])) for r in s['Dihedral Coeffs']}
    if sorted(coeff)!=list(range(1,17)):raise ValueError('expected sixteen XXXII dihedral types')
    for i,row in coeff.items():
        if len(row)!=5:raise ValueError('native five-coefficient dihedral required')
        k,n,phase,wl,wc=row
        if n!=int(n) or n<0 or phase not in (0.,180.):raise ValueError('unsupported XXXII torsion parameters')
        expected=(.5,.8333333333) if i<=8 else (0.,0.)
        if (wl,wc)!=expected:raise ValueError('XXXII endpoint-weight contract violated')
        if i>8 and row[:3]!=coeff[i-8][:3]:raise ValueError('paired torsional types must retain equal K,n,phase')
    adjacency=collections.defaultdict(set)
    for _,typ,i,j in s['Bonds']:
        i,j=int(i),int(j);adjacency[i].add(j);adjacency[j].add(i)
    def distance3(i):
        seen={i};front={i}
        for _ in range(3):front={j for k in front for j in adjacency[k]}-seen;seen|=front
        return front
    expected_pairs={tuple(sorted((i,j))) for i in range(1,173) for j in distance3(i)}
    pairs=collections.defaultdict(list)
    for ident,typ,i,j,k,l in s['Dihedrals']:pairs[tuple(sorted((int(i),int(l))))].append(int(typ))
    if len(s['Atoms'])!=172 or len(s['Bonds'])!=180 or len(s['Dihedrals'])!=432 or len(pairs)!=392:
        raise ValueError('XXXII atom/topology cardinality mismatch')
    if set(pairs)!=expected_pairs:raise ValueError('endpoint pairs differ from graph distance-three pairs')
    if any(sum(typ<=8 for typ in types)!=1 for types in pairs.values()):raise ValueError('each terminal pair needs exactly one weighted dihedral')
    return dict(atoms=172,bonds=180,angles=len(s['Angles']),dihedrals=432,impropers=len(s['Impropers']),endpoint_pairs=392,
        weighted_entries=392,zero_weight_duplicates=40,all_pairs_shortest_bond_distance=3)

def convert(data_path,input_path,output):
    data_path,input_path,output=map(Path,(data_path,input_path,output))
    raw=data_path.read_bytes();commands=input_path.read_bytes()
    if digest(raw)!=DATA_SHA or digest(commands)!=INPUT_SHA:
        raise ValueError('source SHA mismatch: converter supports only the audited original XXXII files')
    text=raw.decode();s=sections(text);topology=verify_topology(s)
    if len(s['Pair Coeffs'])!=11 or any(len(r)!=3 for r in s['Pair Coeffs']):raise ValueError('expected eleven diagonal epsilon/R rows')
    converted=[];current=None
    for line in text.splitlines(keepends=True):
        content=line.split('#')[0].strip()
        if content and content[0].isalpha():current=content
        elif content and current=='Pair Coeffs':
            typ,eps,R=content.split();eps,R=float(eps),float(R);sigma=R/2**(1/6)
            line=f' {typ} {eps:.17g} {sigma:.17g} {(.6*eps):.17g} {sigma:.17g}\n'
        elif content and current=='Dihedral Coeffs':
            typ,k,n,phase,wl,wc=content.split();weight=5/6 if float(wc)>0 else 0.
            line=f' {typ} {k} {n} {phase} {weight:.17g}\n'
        converted.append(line)
    new_data=''.join(converted)
    # Exact source pin makes these replacements controlled, not a generic parser.
    new_commands=commands.decode().replace('dihedral_style  amber','dihedral_style  charmm').replace('pair_style      lj/amber/coul/long 9 10','pair_style      lj/charmm/coul/long 9 10').replace('special_bonds  lj 0.1 0.1 0.1 coul 0.1 0.1 0.1','special_bonds  lj 0 0 0 coul 0 0 0')
    after=sections(new_data)
    for name,rows in s.items():
        if name not in ('Pair Coeffs','Dihedral Coeffs') and after[name]!=rows:raise AssertionError(f'changed {name}')
    if output.exists() and any(output.iterdir()):raise ValueError('output directory must be absent or empty')
    output.mkdir(parents=True,exist_ok=True)
    (output/'lmp.data').write_text(new_data);(output/'in.simple').write_text(new_commands)
    manifest=dict(status='converted_not_engine_validated',scope='audited original XXXII only; no user model substitution',
        source=dict(data=str(data_path.resolve()),input=str(input_path.resolve()),data_sha256=DATA_SHA,input_sha256=INPUT_SHA),
        output_sha256=dict(data=digest(new_data.encode()),input=digest(new_commands.encode())),topology=topology,
        transformation=dict(pair_style='lj/charmm/coul/long 9 10',dihedral_style='charmm',ordinary_epsilon='unchanged',sigma='native R / 2**(1/6)',epsilon14='0.6 * native epsilon14 (equals normal epsilon in supplied data)',sigma14='native R14 / 2**(1/6)',nonzero_dihedral_weight=5/6,zero_dihedral_weight=0.,special_bonds='lj 0 0 0 coul 0 0 0',native_coulomb_weight_snap='0.83 < input < 0.84 -> exact 5/6'),
        preserved=['atom IDs','molecule IDs','atom types','charges','coordinates','cell template','all bonded topology','masses','bond/angle/improper coefficients','torsion K,n,phase','units real','arithmetic mixing','Ewald 1e-6'],
        cell_contract='41x32x51 template retained verbatim; ASE adapter must replace with actual XXXII ARC cell and lifted coordinates without unintended affine double-remap',
        physical_requests=0,qualification='Algebra and original-instruction slices only; full-engine E/F/stress agreement pending',
        evidence='docs/research/native-dihedral-amber-contract.md',stock_source_commit='9c5ab448c78a14fd534619622162ba418d6a1fb1')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(convert(a.data,a.input,a.output),indent=2))
if __name__=='__main__':main()
