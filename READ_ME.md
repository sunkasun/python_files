# Projet : Pricer d'options européennes et américaines, par arbre binomial ou trinomial

## Architecture du code

| Module | Classes | Rôle |
|---|---|---|
| `app` | `PricingApp` | Interface Streamlit, vérifie les dates et limites de mémoire, lance le pricing et orchestre l’affichage. |
| `Market` | `Market` | Stocke S0, le taux, la volatilité, le montant du dividende et sa date. Calcule notamment l’espérance et la variance conditionnelles. |
| `Option` | `Option` | Stocke K, la maturité, le type call/put et le style européen/américain. Calcule les payoffs. |
| `PriceParameters` | `PriceParameters`, `TreeMethod` | Stocke N, la méthode d’arbre, le pruning, son seuil, le moteur et la méthode de valorisation. |
| `Tree` | `Tree`, `TreeNode`, `TreeVectorized` | Calcule les paramètres temporels et construit l’arbre sous forme de liens entre objets ou de tableaux NumPy. |
| `Node` | `Node`, `Trunk` | Représentent les nœuds, leurs voisins et leurs enfants. `Trunk` fournit un repère central pour parcourir les niveaux. |
| `Level` | `LevelNode`, `LevelVectorized` | Construit les transitions d’un niveau au suivant, calcule les probabilités et applique le pruning. |
| `OptionPricer` | `OptionPricerTree` | Calcule les valeurs de l’option par backward, récursion ou pricing vectorisé de plusieurs options. |
| `Black_Scholes` | `Black_Scholes`, `GreeksBS` | Calcule le prix et les grecques (Delta, Gamma, Vega, Rho, Theta, Vomma, Vanna) de Black–Scholes. |
| `Sensibilities` | `GreeksTree`, `OneDimDerivative` | Calcule les sensibilités de l’arbre par différences finies et chocs de paramètres. |
| `Orchestrator` | `Orchestrator` | Relie construction, pricing, Black–Scholes, grecques et benchmarks, puis libère les arbres après usage. |
| `TreeVisualizer` | `TreeVisualizer` | Construit la représentation interactive de l’arbre et de la frontière d’exercice. |
| `Study` | `EtudePricing`, `EtudeComparative`, `EtudePerformances` | Prépare les tableaux, les indicateurs et les graphiques affichés. |

## Architecture fonctionnelle du pricer
1.	L’utilisateur renseigne les paramètres dans le menu latéral.
2.	PricingApp instancie les objets Market, Option et PriceParameters. Un dividende situé hors de l’intervalle (date de pricing, maturité] est neutralisé.
3.	La méthode Orchestrator.build sélectionne TreeNode ou TreeVectorized selon le moteur choisi.
4.	La classe Tree calcule T, Δt, le facteur d’actualisation et les paramètres de grille à chaque pas.
5.	LevelNode ou LevelVectorized construit les nœuds et les probabilités de transition, en appliquant éventuellement le pruning.
6.	OptionPricerTree calcule les payoffs à maturité puis remonte l’arbre jusqu’à la racine.
7.	Orchestrator conserve les résultats utiles et, si nécessaire, reconstruit une représentation vectorisée de l’arbre pour les graphiques.
8.	Les classes d’étude calculent les indicateurs et affichent les tableaux et graphiques dans Streamlit.