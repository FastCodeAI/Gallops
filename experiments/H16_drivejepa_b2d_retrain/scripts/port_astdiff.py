"""What each ported module changed against the file as it ran: ast.unparse of both, docstrings dropped, unified diff."""
import ast, difflib, pathlib, sys, runpy
sys.argv=['x']
ns = {}
src = open(pathlib.Path(__file__).with_name('port_edits.py')).read().split("DST.mkdir")[0]
exec(src, ns)
FILES, DST = ns['FILES'], ns['DST']
def norm(text):
    t = ast.parse(text)
    for n in ast.walk(t):
        if isinstance(n,(ast.Module,ast.FunctionDef,ast.ClassDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):
            n.body = n.body[1:] or [ast.Pass()]
    # one statement per line, imports sorted out of the way of the comparison
    return ast.unparse(t).split('\n')
tot = 0
for name,(srcf,_) in FILES.items():
    a, b = norm(srcf.read_text()), norm((DST/name).read_text())
    d = [l for l in difflib.unified_diff(a,b,lineterm='',n=0) if not l.startswith(('---','+++','@@'))]
    print(f'### {name} <- {srcf.name}: {len(a)} lines, {len(d)} changed')
    for l in d: print('  ', l[:170])
    tot += len(d)
print('TOTAL', tot)
