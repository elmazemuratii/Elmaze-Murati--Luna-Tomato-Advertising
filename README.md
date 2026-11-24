--Luna-Tomato-Advertising

1. Problemi
Mesazhi "Tomatoes for sale" kthehet në sinjal Morse binar (2710 mostra, dt = 0.01036 ditë). Çdo anije në një orbitë periodike
(DRO, Lyapunov ose Axial) prodhon një "kuti" okultimi me gjerësi 0.1036 ditë (= 10 mostra) çdo periodë orbitale
(Lyapunov/Axial kanë dy kuti për periodë). Sinjali i rindërtuar është shuma e kutive e prerë në [0,1].
Synimi: sa më pak anije, me MSE ≤ 0.05 (pra ≤ 135 mostra të gabuara nga 2710). Maksimumi është 5 anije për orbitë.

2. Formati i zgjidhjes
Vektori x ka gjatësi N + 5N (N = 188 + 291 + 300 = 779): së pari numri i anijeve për orbitë (0–5), pastaj 5 faza për orbitë në [0, 2π).
Rendi i orbitave: DRO, Lyapunov, Axial.

3. Algoritmi (lta/solver.py)
Mjedisi pa heyoka (lta/problem.py): riimplementim i saktë i fitness zyrtar; nuk duhet integrim numerik, sepse DB ka frekuencat.
Maskat: çdo kuti mbulon saktësisht 10 mostra, kështu gabimi = numri i mostrave ku (mbulimi>0) ≠ objektivi.
Greedy: me shuma prefiksore, secili kandidat (orbitë, fazë) vlerësohet në O(kuti). Jep 41 anije, gabim 404.
LNS / Simulated Annealing: hiq 1–3 anije, rifut më të mirat nga skanimi i një nënbashkësie orbitash, pranim Metropolis.
Shrink: kur zgjidhja është e pranueshme, hiqet anija me humbjen më të vogël (K → K−1) dhe LNS riparon gabimin.
Verifikim: verify_official.py e provon zgjidhjen me lta_udp.py origjinal të ESA-s.
4. Rezultati
Anije: 35 (LYAP: 25, AXIAL: 7, DRO: 3)
MSE: 0.049815 (prag 0.05), objektivi = 35
Konfirmuar me kodin zyrtar të ESA-s: feasible=True.
Përpjekjet për 34 anije nuk kaluan brenda kohës që kam provuar; mund të provohet me --time më të gjatë ose seed të tjerë.
5. Vërejtje
Në evaluatorin zyrtar, kutia e dytë e Lyapunov/Axial përdor (ph+π)/f (radianë të përzier me kohë të pa-dimensionuar), ndryshe nga
përshkrimi (okultim në π). Zgjidhja ndjek formulën zyrtare, sepse ajo përcakton pikët. Kjo e bën pozicionin e kutisë së dytë shumë të
ndjeshëm ndaj fazës, prandaj rrjeta e fazave është e hollë (16384 pika).
Në vizualizimin 3D, pozicioni i anijes animohet me kutinë e parë; flash-i i okultimit merr parasysh të dyja kutitë.

6. Si ta përdorësh në PyCharm
pip install -r requirements.txt
python run_solver.py --time 1700 --per-k 45 --seed 1     # kërkimi (ruan results/best_fleet.json)
python run_solver.py --start results/best_fleet.json --seed 7 --time 3000   # vazhdim me seed tjetër
python make_submission.py            # -> results/solution.json
python verify_official.py results/solution.json "<rruga>/SpOC4/Challenge 3 Luna Tomato Advertising/lta_udp.py"
python export_3d.py                  # -> results/lta_3d.html (hape në shfletues)
Dorëzimi: me _submission_helper të ESA-s,
create_submission("spoc-4-luna-tomato-advertising", "tie-breaker", x, "solutions/my_submission.json"),
ku x është fusha x në results/solution.json.

7. Struktura
lta/problem.py evaluatori · lta/solver.py greedy/LNS/shrink · lta/orbits.py CR3BP (scipy) · lta/viewer_template.html skena three.js ·
data/ bazat e orbitave të ESA-s · results/ zgjidhja dhe faqja 3D.
