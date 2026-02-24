<?php

namespace App\Services;

use App\Models\AnomalieFacture;
use App\Models\DossierFacture;
use App\Models\Facture;
use App\Models\Structure;
use App\Models\ValidationActivity;
use Carbon\Carbon;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Log;
use Madnest\Madzipper\Facades\Madzipper;

class VerificationService
{

    public static function lancerVerification(Facture $facture, string $decisionRejet = 'a_corriger'){
        // Déterminer la décision en fonction du statut actuel
        $statutActuel = $facture->statut_validation;
        $decisionRejet = $statutActuel == null ? 'a_corriger' : 'rejetee';

        // Appels aux fonctions de vérification avec la nouvelle logique
        self::verifierCompletude($facture, $decisionRejet);
        self::verifierIncoherenceSexe($facture, $decisionRejet);
        self::verifierIncoherenceFormationSanitaire($facture, $decisionRejet);
        self::verifierHospitalisation($facture, $decisionRejet);
        self::verifierPrestationEnfant($facture, $decisionRejet);
        self::verifierHospitalisationEtEvacuation($facture, $decisionRejet);
        self::verifierQuantiteProduit($facture, $decisionRejet);
        self::verifierQuantiteActe($facture, $decisionRejet);
        self::verifierQuantiteExamen($facture, $decisionRejet);
        self::verifierExamenHospitalisation($facture, $decisionRejet);
        self::verifierMontantEvacuation($facture, $decisionRejet);
        self::verifierActesNonCumulatifs($facture, $decisionRejet);
        self::verifierActesAssocies($facture, $decisionRejet);
        self::verifierActesExclusifs($facture, $decisionRejet);
        self::verifierIncoherenceEvacuation($facture, $decisionRejet);
        self::verifierIncoherenceDateEntree($facture, $decisionRejet);
        self::verifierIncoherenceDateSortie($facture, $decisionRejet);
        self::verifierIncoherenceDatePrescriptionMed($facture, $decisionRejet);
        self::verifierIncoherenceDatePrescriptionExam($facture, $decisionRejet);
        self::verifierChevauchementHospitalisation($facture, $decisionRejet);
        self::verifierIncoherenceAntibiotique($facture, $decisionRejet);

        // Vérification des décisions d'anomalies
        $status = "validee";

        $rejectedExists = AnomalieFacture::where('facture_id', $facture->id)
            ->where('decision', 'rejetee')
            ->exists();

        if ($rejectedExists) {
            $status = "rejetee";
        } else {
            $correctionsExist = AnomalieFacture::where('facture_id', $facture->id)
                ->where('decision', 'a_corriger')
                ->exists();

            if ($correctionsExist) {
                $status = "a_corriger";
            }
        }

        // Mise à jour de la facture
        $facture->update([
            'statut_validation' => $status,
            'last_verification_at' => now(),
        ]);

        // Ajouter dans l'historique
        ValidationActivity::create([
            'facture_id' => $facture->id,
            'status' => $status,
            'anomalies' => $facture->anomalies ? json_encode($facture->anomalies) : null,
            'verified_by' => 'System',
            'verified_at' => now()
        ]);
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 1 => Forme
     * Factures contenant des valeurs manquantes sur au moins une variable obligatoire
     */
    private static function verifierCompletude(Facture $facture, string $decisionRejet = 'a_corriger'): void
    {

        // Liste des champs obligatoires à vérifier
        $champsObligatoires = [
            'age_patient',
            'distance_village',
            'registre_number',
            'nom_patient',
            'prestation',
            'provenance',
            'sexe',
            'type_prestation',
            'nom_gerant',
            'visit_date',
        ];

        $champsManquants = [];
        $typeFs = self::getTypeFs($facture->formation_sanitaire);

        foreach ($champsObligatoires as $champ) {
            // Ignorer la vérification de distance_village pour CHR et CHU
            if ($champ == 'distance_village' && in_array($typeFs, ['CHR', 'CHU'])) {
                continue;
            }

            if (is_null($facture->$champ) || trim($facture->$champ) == '') {
                $champsManquants[] = $champ;
            }
        }

        if (!empty($champsManquants)) {
            $descriptionAnomalie = 'Complétude des données individuelles';
            $decision = $decisionRejet;
            $commentaire = 'Absence de: ' . implode(', ', $champsManquants);

            self::creerAnomalie(
                $facture,
                "Forme",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 2 et 3
     *  - Sexe masculin bénéficiant de : accouchements, interventions obstétricales,
     *    soins pendant la grossesse, PF (hormis vasectomie et condom masculin),
     *    dépistage + traitement des lésions précancéreuses
     *  - Sexe féminin bénéficiant de : vasectomie
     */
    private static function verifierIncoherenceSexe(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        // Liste des prestations et types de prestations incohérents pour un patient de sexe masculin
        $prestationsIncoherentes = [
            'accouchement et intervention obstétricale',
            'soins pendant la grossesse',
            'planification familiale',
            'dépistage + traitement des lésions précancéreuses'
        ];

        // Vérification des incohérences
        if ($facture->sexe == 'male') {
            if (in_array(strtolower($facture->prestation), $prestationsIncoherentes) &&
                !(strtolower($facture->type_prestation) == 'vasectomie' || strtolower($facture->type_prestation) == 'condom masculin')) {

                $descriptionAnomalie = 'Qualité des données transmises (irrégularités, incohérences)';
                $decision = $decisionRejet;
                $commentaire = "L’acte ne correspond pas avec le sexe du patient";

                self::creerAnomalie(
                    $facture,
                    "Fond",
                    $descriptionAnomalie,
                    $decision,
                    $commentaire
                );
            }
        } elseif ($facture->sexe == 'female' && strtolower($facture->type_prestation) == 'vasectomie') {
            // Cas où une femme bénéficie d'une vasectomie
            $descriptionAnomalie = 'Qualité des données transmises (irrégularités, incohérences)';
            $decision = $decisionRejet;
            $commentaire = "L’acte ne correspond pas avec le sexe du patient";


            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }


    /**
     * @param Facture $facture
     * @return void
     *
     * Algo 4
     * CSPS ayant réalisé : césarienne, laparotomie pour rupture utérine, laparotomie pour GEU,
     * cure de fistules obstétricales, ligature des trompes, vasectomie,
     * Résection à l'anse diathermique, cryothérapie
     */
    private static function verifierIncoherenceFormationSanitaire(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        // Liste des actes non éligibles pour un CSPS ou DISPENSAIRE
        $actesNonEligibles = [
            strtolower('Laparotomie (GEU, RU, autres)'),
            strtolower('césarienne'),
            strtolower('Cure de fistule recto vaginale'),
            strtolower('Cure de fistule vésico-vaginale'),
        ];

        $typeFs = self::getTypeFs($facture->formation_sanitaire);

        // Vérification si le nom de la formation sanitaire contient CSPS ou DISPENSAIRE
        if ( $typeFs == 'CSPS' || $typeFs == 'DISPENSAIRE') {
            if(!is_null($facture->liste_acte)){

                $listeActes = json_decode($facture->liste_acte, true);

                foreach ($listeActes as $acte) {
                    if (in_array($acte['description'], $actesNonEligibles)) {
                        $descriptionAnomalie = 'Qualité des données transmises (irrégularités, incohérences)';
                        $decision = $decisionRejet;
                        $commentaire = "Prestation non éligible au niveau CSPS ou DISPENSAIRE";


                        self::creerAnomalie(
                            $facture,
                            "Fond",
                            $descriptionAnomalie,
                            $decision,
                            $commentaire
                        );

                    }
                }
            }

        }
    }

    /**
     * @param Facture $facture
     * @return void
     *
     * Algo 5 et 7
     *  - Bénéficiaires PF bénéficiant des biens/services : hospitalisation
     *  - Patient ambulatoire bénéficiant des biens/services :
     * hospitalisation
     *  -
     */
    public static function verifierHospitalisation(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        $prestationsNonEligibles = [
            strtolower('planification familiale'),
        ];

        $typesPrestationsNonEligibles = [
            strtolower('soins curatifs en ambulatoire'),
        ];

        // Vérifier si la prestation ou le type de prestation est non éligible et s'il y a une hospitalisation
        if (
            in_array(strtolower($facture->prestation), $prestationsNonEligibles) ||
            in_array(strtolower($facture->type_prestation), $typesPrestationsNonEligibles)
        ) {
            if (!is_null($facture->type_hospitalisation)) {
                $descriptionAnomalie = "Surfacturation";
                $decision = $decisionRejet;
                $commentaire = "Prestation non éligible pour la " . ucfirst($facture->prestation) . " avec type de prestation " . ucfirst($facture->type_prestation);

                self::creerAnomalie(
                    $facture,
                    "Fond",
                    $descriptionAnomalie,
                    $decision,
                    $commentaire
                );
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 6
     * Bénéficiaires
     * Dépistage+traitement des lésions précancéreuses bénéficiant des biens/services : hospitalisation et évacuation
     */
    public static function verifierHospitalisationEtEvacuation(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        $prestationsNonEligibles = [
            'dépistage + traitement des lésions précancéreuses',
        ];

        // Vérifier si la prestation est non éligible et s'il y a une hospitalisation et une évacuation
        if (
            in_array(strtolower($facture->prestation), $prestationsNonEligibles) &&
            !is_null($facture->type_hospitalisation) &&
            $facture->cout_evacuation > 0
        ) {
            $descriptionAnomalie = "Surfacturation";
            $decision = $decisionRejet;
            $commentaire = "Bien et service non éligible pour la catégorie de patient";
            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * ALGO 9
     *  Cibles de moins de 9 ans bénéficiant de :
     * accouchements+interventions obstétricales, soins pendant la grossesse, PF,
     * dépistage+traitement des lésions précancéreuses
     */
    public static function verifierPrestationEnfant(Facture $facture, string $decisionRejet = 'a_corriger'){

        $prestationsNonEligibles = [
            'soins pendant la grossesse',
            'planification familiale',
            'accouchement et intervention obstétricale',
            'dépistage + traitement des lésions précancéreuses',
        ];

        // Extraire l'âge en années
        $age = self::convertirAgeEnAnnees($facture->age_patient);

        // Vérifier si l'âge est inférieur à 9 ans et si la prestation est non éligible
        if ($age < 9 && in_array(strtolower($facture->prestation), $prestationsNonEligibles)) {
            $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
            $decision = $decisionRejet;
            $commentaire = "Bien et service non éligible pour la catégorie de patient";

            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }

    /**
     * @param Facture $facture
     * @return void
     *  Algo 10
     *
     * Facture contenant un médicament dont la quantité >=100
     *
     */
    public static function verifierQuantiteProduit(Facture $facture, string $decisionRejet = 'a_corriger'){

        if(!is_null($facture->liste_produit)){
            $listeProduits = json_decode($facture->liste_produit, true);

            // Vérifier si un produit a une quantité > 100
            foreach ($listeProduits as $produit) {
                if (isset($produit['quanite_prod']) && $produit['quanite_prod'] > 100) {
                    $descriptionAnomalie = "Surfacturation";
                    $decision = $decisionRejet;
                    $commentaire = "Quantité anormale de médicament";

                    self::creerAnomalie(
                        $facture,
                        "Fond",
                        $descriptionAnomalie,
                        $decision,
                        $commentaire
                    );
                }
            }
        }

    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 11
     *
     * Facture contenant des actes dont la quantité > 1 a l’exception des forfaits journaliers
     */
    public static function verifierQuantiteActe(Facture $facture, string $decisionRejet = 'a_corriger'){

        if(!is_null($facture->liste_acte)){
            $listeActes = json_decode($facture->liste_acte, true);

            // Vérifier si un acte a une quantité > 1
            foreach ($listeActes as $acte) {
                //On verifie que l'acte n'est pas dans la liste des forfaits journaliers
                if(!array_key_exists($acte['liste_act'], SettingService::getForfaits())){

                    if (isset($acte['quanite_act']) && $acte['quanite_act'] > 1) {
                        $descriptionAnomalie = "Surfacturation";
                        $decision = $decisionRejet;
                        $commentaire = "Quantité anormale d’acte Réduire la quantité à 1";

                        self::creerAnomalie(
                            $facture,
                            "Fond",
                            $descriptionAnomalie,
                            $decision,
                            $commentaire
                        );
                    }
                }
            }
        }

    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 12
     *
     * Facture contenant des jours d'hospitalisation >= 10
     */
    /*public static function verifierNbreJourHospitalisation(Facture $facture, string $decisionRejet = 'a_corriger'){

        if(!is_null($facture->type_hospitalisation)){
            $hospitalisations = json_decode($facture->type_hospitalisation, true);

            foreach ($hospitalisations as $hospitalisation) {
                if (isset($hospitalisation['nombre_jour']) && $hospitalisation['nombre_jour'] >= 10) {

                    self::creerAnomalie(
                        $facture,
                        "Fond",
                        "Surfacturation",
                        "a_corriger",
                        "Nombre de jour d'hospitalisation anormal"
                    );
                }
            }
        }
    }*/

    /**
     * @param Facture $facture
     * @return void
     *  Algo 13
     * FIS des patients en ambulatoire contenant des examens complémentaires dont la quantité > 1 par type d’examen
     */
    public static function verifierQuantiteExamen(Facture $facture, string $decisionRejet = 'a_corriger'){

        if (strtolower($facture->type_prestation) == 'soins curatifs en ambulatoire') {

            if(!is_null($facture->liste_examen)){
                $listeExamen = json_decode($facture->liste_examen, true);

                foreach ($listeExamen as $examen) {
                    if (isset($examen['quantite_ex']) && $examen['quantite_ex'] > 1) {
                        $descriptionAnomalie = "Surfacturation";
                        $decision = $decisionRejet;
                        $commentaire = "Quantité anormale d'examen. Réduire la quantité à 1";

                        self::creerAnomalie(
                            $facture,
                            "Fond",
                            $descriptionAnomalie,
                            $decision,
                            $commentaire
                        );
                    }
                }
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     *
     * Algo 14
     *
     * Facture des patients en hospitalisation contenant des examens complémentaires dont la quantité > 1
     * pour les examens devant être réalisés une seule fois au cours de l'hospitalisation (groupage sanguin,électrophorèse de l'hémoglobine …)
     *
     */
    public static function verifierExamenHospitalisation(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        // Liste des examens qui doivent être réalisés une seule fois
        $examensUnique = [
            "groupe sanguin",
            "électrophorèse de l'hémoglobine"
        ];

        // Vérifier si la facture concerne une hospitalisation
        if (!is_null($facture->type_hospitalisation)) {
            if(!is_null($facture->liste_examen)){
                $listeExamen = json_decode($facture->liste_examen, true);

                foreach ($listeExamen as $examen) {
                    // Vérifier si l'examen est dans la liste des examens uniques et si la quantité est supérieure à 1
                    if (in_array(strtolower($examen['nom_examen']), $examensUnique) && $examen['quanite_ex'] > 1) {
                        $descriptionAnomalie = "Surfacturation";
                        $decision = $decisionRejet;
                        $commentaire = "Quantité anormale d'examen. Réduire la quantité à 1";

                        self::creerAnomalie(
                            $facture,
                            "Fond",
                            $descriptionAnomalie,
                            $decision,
                            $commentaire
                        );
                    }
                }
            }

        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 15
     * Facture contenant des montants d'évacuation >= 120000 F
     */
    public static function verifierMontantEvacuation(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        // Vérification si le montant d'évacuation est supérieur ou égal à 120000
        if ($facture->cout_evacuation >= 120000) {
            $descriptionAnomalie = "Surfacturation";
            $decision = $decisionRejet;
            $commentaire = "Montant d'évacuation anormal";

            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }


    /**
     * @param Facture $facture
     * @return void
     *
     * Algo 16 et 18
     * Facture contenant à la fois des actes principaux et des actes qui leur sont accessoires :
     * "Accouchement" = acte principal intégrant les actes accessoires de "Délivrance Artificielle + Révision Utérine" ; "Tension artérielle" ;
     * "injection" ; "consultation"
     * "Consultation" = acte principal intégrant les actes accessoires de "Tension artérielle",  "injection"
     */
    public static function verifierActesNonCumulatifs(Facture $facture, string $decisionRejet = 'a_corriger'){
        // Définir les actes principaux et leurs actes accessoires
        $actesAccessoires = [
            'accouchement' => [
                'délivrance artificielle + révision utérine',
                'tension artérielle',
                'injection',
                'consultation'
            ],
            'consultation' => [
                'tension artérielle',
                'injection'
            ]
        ];

        if(!is_null($facture->liste_acte)){
            $listeActes = json_decode($facture->liste_acte, true);

            // Stocker les actes trouvés
            $actesTrouves = [];

            foreach ($listeActes as $acte) {
                $actesTrouves[] = $acte['description'];
            }

            // Vérifier si un acte principal est dans la liste des actes
            foreach ($actesAccessoires as $actePrincipal => $accessoires) {
                if (in_array(strtolower($actePrincipal), $actesTrouves)) {
                    // Vérifier si des actes accessoires sont également présents
                    foreach ($accessoires as $accessoire) {
                        if (in_array(strtolower($accessoire), $actesTrouves)) {
                            // Créer une anomalie pour surfacturation (actes non cumulatifs)
                            $descriptionAnomalie = "Surfacturation";
                            $decision = $decisionRejet;
                            $commentaire = "Actes non cumulatifs. Supprimer les actes accessoires";

                            self::creerAnomalie(
                                $facture,
                                "Fond",
                                $descriptionAnomalie,
                                $decision,
                                $commentaire
                            );
                        }
                    }
                }
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 17
     * FIS contenant à la fois des actes associés suivants : "Episiotomie » ; "Suture"
     */
    public static function verifierActesAssocies(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        $actesRecherche = ['episiotomie', 'suture'];
        $actesTrouves = [];

        if (!is_null($facture->liste_acte)) {
            $listeActe = json_decode($facture->liste_acte, true);

            // Parcourir la liste des actes et vérifier si on trouve "Episiotomie" et "Suture"
            foreach ($listeActe as $acte) {
                if (in_array(strtolower($acte['description']), $actesRecherche)) {
                    $actesTrouves[] = $acte['description'];

                    if (count($actesTrouves) == 2) {
                        break;
                    }
                }
            }
        }

        // Vérification si on a trouvé à la fois "Episiotomie" et "Suture"
        if (count($actesTrouves) == 2) {
            $descriptionAnomalie = "Surfacturation";
            $decision = $decisionRejet;
            $commentaire = "Actes non cumulatifs. Supprimer les actes accessoires";

            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 19
     * Facture contenant à la fois des actes mutuellement exclusifs : "Accouchement normal",
     * "Accouchement assisté", "Césarienne"
     */
    public static function verifierActesExclusifs(Facture $facture, string $decisionRejet = 'a_corriger'){
        $actesRecherche = ['accouchement normal', 'accouchement assisté', 'césarienne'];
        $actesTrouves = [];

        if (!is_null($facture->liste_acte)) {
            $listeActe = json_decode($facture->liste_acte, true);

            foreach ($listeActe as $acte) {
                if (in_array(strtolower($acte['description']), $actesRecherche)) {
                    $actesTrouves[] = $acte['description'];

                    if (count($actesTrouves) == 2 || count($actesTrouves) == 3) {
                        break;
                    }
                }
            }
        }

        if (count($actesTrouves) == 2 || count($actesTrouves) == 3) {
            $descriptionAnomalie = "Surfacturation";
            $decision = "rejetee";
            $commentaire = "Actes non cumulatifs. Supprimer les actes accessoires";

            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 20
     * Facture des patients dont le mode de sortie est différent de
     * "Evacué" et contenant un montant d'évacuation
     */
    public static function verifierIncoherenceEvacuation(Facture $facture, string $decisionRejet = 'a_corriger'){

        if (strtolower($facture->mode_sortie) !== 'evacuation' && (float) $facture->cout_evacuation > 0) {

            $descriptionAnomalie = "Surfacturation";
            $decision = "rejetee";
            $commentaire = "Supprimer le montant de l'évacuation";

            self::creerAnomalie(
                $facture,
                "Fond",
                $descriptionAnomalie,
                $decision,
                $commentaire
            );
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 21
     * Facture dont la date d'entrée dans la formation sanitaire est postérieure à la date de saisie de la FIS
     */
    public static function verifierIncoherenceDateEntree(Facture $facture, string $decisionRejet = 'a_corriger')
    {

        if(!is_null($facture->date_entree)){
            $dateEntree = Carbon::parse($facture->date_entree);
            $createdAt = Carbon::parse($facture->created_at);

            if ($dateEntree->gt($createdAt)) {
                $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
                $decision = $decisionRejet;
                $commentaire = "Facture saisie avant date d'entrée";

                self::creerAnomalie(
                    $facture,
                    "Fond",
                    $descriptionAnomalie,
                    $decision,
                    $commentaire
                );
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 22
     *  Facture dont la date de sortie de la formation sanitaire est postérieure à la date de saisie de la Facture
     */
    public static function verifierIncoherenceDateSortie(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        if(!is_null($facture->date_sortie)){
            $dateSortie = Carbon::parse($facture->date_sortie);
            $createdAt = Carbon::parse($facture->created_at);

            if ($dateSortie->gt($createdAt)) {
                $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
                $decision = $decisionRejet;
                $commentaire = "Date de sortie est avant la date d’entrée";

                self::creerAnomalie(
                    $facture,
                    "Fond",
                    $descriptionAnomalie,
                    $decision,
                    $commentaire
                );
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 23
     * Facture dont la date de prescription des médicaments dans la formation sanitaire est postérieure à la date de saisie de la Facture
     */
    public static function verifierIncoherenceDatePrescriptionMed(Facture $facture, string $decisionRejet = 'a_corriger'): void
    {
        // verifier que des produits sont disponibles
        if(!is_null($facture->liste_produit)){
            $listeProduit = json_decode($facture->liste_produit, true);

            foreach ($listeProduit as $produit) {
                // Vérifier si la date de prescription du produit n'est pas null
                if(isset($produit->date_prescription) && $produit->date_prescription !== 'null'){
                    $datePresc = Carbon::parse($produit->date_prescription);
                    $createdAt = Carbon::parse($facture->created_at);
                    if ($datePresc->gt($createdAt)) {
                        $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
                        $decision = $decisionRejet;
                        $commentaire = "Date de prescription est avant la date d'entrée";

                        self::creerAnomalie(
                            $facture,
                            "Fond",
                            $descriptionAnomalie,
                            $decision,
                            $commentaire
                        );
                    }
                }
                //break;
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Algo 24
     * Facture dont la date de prescription des examens dans la formation sanitaire est postérieure à la date de saisie de la Facture
     */
    public static function verifierIncoherenceDatePrescriptionExam(Facture $facture, string $decisionRejet = 'a_corriger'): void
    {
        // verifier que des examens sont disponibles
        if(!is_null($facture->liste_examen)){
            $listeExamen = json_decode($facture->liste_examen, true);

            foreach ($listeExamen as $examen) {
                // Vérifier si la date de prescription de l'examen n'est pas null
                if(isset($examen->date_prescription) && $examen->date_prescription !== 'null'){
                    $datePresc = Carbon::parse($examen->date_prescription);
                    $createdAt = Carbon::parse($facture->created_at);
                    if ($datePresc->gt($createdAt)) {
                        $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
                        $decision = $decisionRejet;
                        $commentaire = "Date de prescription est avant la date d'entrée";

                        self::creerAnomalie(
                            $facture,
                            "Fond",
                            $descriptionAnomalie,
                            $decision,
                            $commentaire
                        );
                    }
                }
            }
        }
    }


    // /**
    //  * Chevauchement hospitalisation
    //  */
    // public static function verifierChevauchementHospitalisation(Facture $facture, string $decisionRejet = 'a_corriger'){
    //     $typeFs = self::getTypeFs($facture->formation_sanitaire);

    //     if($typeFs == 'CHR' || $typeFs == 'CHU'){
    //         // Vérifier si la facture concerne une hospitalisation
    //         if (!is_null($facture->type_hospitalisation)) {

    //                 $listeHospitalisation = json_decode($facture->type_hospitalisation, true);

    //                 foreach ($listeHospitalisation as $hospitalisation) {
    //                     //S'il ya une seul hospitalisation, pas besoin de vérifier le chevauchement
    //                     if (count($listeHospitalisation) <= 1) {
    //                         return;
    //                     }

    //                     $chevauchementDetecte = false;

    //                     // Comparons chaque hospitalisation avec les autres
    //                     for ($i = 0; $i < count($listeHospitalisation); $i++) {
    //                         $hospitalisation1 = $listeHospitalisation[$i];
    //                         $dateEntree1 = Carbon::parse($hospitalisation1['date_entree']);
    //                         $dateSortie1 = Carbon::parse($hospitalisation1['date_sortie']);

    //                         // Comparons avec les hospitalisations suivantes
    //                         for ($j = $i + 1; $j < count($listeHospitalisation); $j++) {
    //                             $hospitalisation2 = $listeHospitalisation[$j];
    //                             $dateEntree2 = Carbon::parse($hospitalisation2['date_entree']);
    //                             $dateSortie2 = Carbon::parse($hospitalisation2['date_sortie']);

    //                             // Vérifions le chevauchement
    //                             if (
    //                                 // Cas 1: dateEntree1 est dans la période de hospitalisation2
    //                                 ($dateEntree1->between($dateEntree2, $dateSortie2)) ||
    //                                 // Cas 2: dateSortie1 est dans la période de hospitalisation2
    //                                 ($dateSortie1->between($dateEntree2, $dateSortie2)) ||
    //                                 // Cas 3: hospitalisation2 est complètement incluse dans hospitalisation1
    //                                 ($dateEntree2->between($dateEntree1, $dateSortie1)) ||
    //                                 ($dateSortie2->between($dateEntree1, $dateSortie1))
    //                             ) {
    //                                 $chevauchementDetecte = true;
    //                                 $details = sprintf(
    //                                     "Chevauchement détecté entre %s (%s au %s) et %s (%s au %s)",
    //                                     $hospitalisation1['type_hospi'],
    //                                     $hospitalisation1['date_entree'],
    //                                     $hospitalisation1['date_sortie'],
    //                                     $hospitalisation2['type_hospi'],
    //                                     $hospitalisation2['date_entree'],
    //                                     $hospitalisation2['date_sortie']
    //                                 );

    //                                 $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
    //                                 $decision = $decisionRejet;
    //                                 $commentaire = "Date de sortie est avant la date d’entrée";

    //                                 self::creerAnomalie(
    //                                     $facture,
    //                                     "Fond",
    //                                     $descriptionAnomalie,
    //                                     $decision,
    //                                     $details
    //                                 );

    //                                 // On peut sortir des boucles dès qu'un chevauchement est détecté
    //                                 break 2;
    //                             }
    //                         }
    //                     }

    //                 }
    //         }
    //     }
    // }

    /**
     * Vérifier le chevauchement d'hospitalisations
     * Règle: Même date entrée/sortie = 1 jour, 1 jour de différence = 1 jour
     */
    public static function verifierChevauchementHospitalisation(Facture $facture, string $decisionRejet = 'a_corriger')
    {
        $typeFs = self::getTypeFs($facture->formation_sanitaire);

        if ($typeFs == 'CHR' || $typeFs == 'CHU') {
            if (!is_null($facture->type_hospitalisation)) {
                $listeHospitalisation = json_decode($facture->type_hospitalisation, true);

                if (count($listeHospitalisation) <= 1) {
                    return;
                }

                for ($i = 0; $i < count($listeHospitalisation); $i++) {
                    $hospitalisation1 = $listeHospitalisation[$i];
                    $dateEntree1 = Carbon::parse($hospitalisation1['date_entree'])->startOfDay();
                    $dateSortie1 = Carbon::parse($hospitalisation1['date_sortie'])->endOfDay();

                    for ($j = $i + 1; $j < count($listeHospitalisation); $j++) {
                        $hospitalisation2 = $listeHospitalisation[$j];
                        $dateEntree2 = Carbon::parse($hospitalisation2['date_entree'])->startOfDay();
                        $dateSortie2 = Carbon::parse($hospitalisation2['date_sortie'])->endOfDay();

                        // Vérifier le chevauchement réel
                        // Un chevauchement existe si les périodes se superposent
                        // Mais on doit tenir compte que même date = 1 jour et 1 jour de diff = 1 jour

                        // Cas 1: Les périodes se chevauchent vraiment
                        $chevauchement = false;

                        // Si la date de sortie de l'une est >= date d'entrée de l'autre
                        // ET la date d'entrée de l'une est <= date de sortie de l'autre
                        if ($dateSortie1->gte($dateEntree2) && $dateEntree1->lte($dateSortie2)) {
                            // Il y a chevauchement potentiel
                            // Mais on doit exclure le cas où elles sont consécutives

                            // Calculer si c'est vraiment un chevauchement ou juste des jours consécutifs
                            // Si sortie1 = entrée2 ou sortie2 = entrée1, ce sont des jours consécutifs, pas un chevauchement

                            $sortie1SansHeure = Carbon::parse($hospitalisation1['date_sortie'])->startOfDay();
                            $entree2SansHeure = Carbon::parse($hospitalisation2['date_entree'])->startOfDay();
                            $sortie2SansHeure = Carbon::parse($hospitalisation2['date_sortie'])->startOfDay();
                            $entree1SansHeure = Carbon::parse($hospitalisation1['date_entree'])->startOfDay();

                            // Vérifier si ce sont des périodes consécutives (sortie = entrée du suivant)
                            $sontConsecutives = $sortie1SansHeure->equalTo($entree2SansHeure) ||
                                            $sortie2SansHeure->equalTo($entree1SansHeure);

                            // Vérifier si sortie de l'un est le jour suivant de l'entrée de l'autre
                            $jourConsecutif = $sortie1SansHeure->copy()->addDay()->equalTo($entree2SansHeure) ||
                                            $sortie2SansHeure->copy()->addDay()->equalTo($entree1SansHeure);

                            if (!$sontConsecutives && !$jourConsecutif) {
                                $chevauchement = true;
                            }
                        }

                        if ($chevauchement) {
                            $details = sprintf(
                                "Chevauchement détecté entre %s (%s au %s) et %s (%s au %s)",
                                $hospitalisation1['type_hospi'],
                                $hospitalisation1['date_entree'],
                                $hospitalisation1['date_sortie'],
                                $hospitalisation2['type_hospi'],
                                $hospitalisation2['date_entree'],
                                $hospitalisation2['date_sortie']
                            );

                            $descriptionAnomalie = "Qualité des données transmises (irrégularités, incohérences)";
                            $decision = $decisionRejet;

                            self::creerAnomalie(
                                $facture,
                                "Fond",
                                $descriptionAnomalie,
                                $decision,
                                $details
                            );
                            return;
                        }
                    }
                }
            }
        }
    }

    /**
     * @param Facture $facture
     * @return void
     * Facture niveau CSPS/CM contenant plus de 2 antibiotiques : (liste antibiotique doit être paramétrable)
     */
    public static function verifierIncoherenceAntibiotique(Facture $facture, string $decisionRejet = 'a_corriger'): void{
        $typeFs = self::getTypeFs($facture->formation_sanitaire);

        if($typeFs == 'CSPS' || $typeFs == 'CM'){

            if(!is_null($facture->liste_produit)){
                $listeProduits = json_decode($facture->liste_produit, true);

                $antibiotique = [];
                foreach ($listeProduits as $produit) {
                    if(array_key_exists($produit['liste_prod'], SettingService::getAntibiotiques())){
                        $antibiotique[] = $produit;
                    }
                }
                if(count($antibiotique) > 2){
                    $descriptionAnomalie = "Interdiction";
                    $decision = $decisionRejet;
                    $commentaire = "Non respect de la norme de prescription  en antibiotiques";

                    self::creerAnomalie(
                        $facture,
                        "Fond",
                        $descriptionAnomalie,
                        $decision,
                        $commentaire
                    );
                }
            }
        }
    }


    public static function exportFacturesToJsonAndZip($dossierId)
    {
        $dossier = DossierFacture::find($dossierId);

        if (!$dossier) {
            Log::error("Dossier introuvable : {$dossierId}");
            return ['success' => false, 'message' => 'Dossier introuvable'];
        }

        try {
            // Créer le dossier s'il n'existe pas
            $directory = storage_path('app/factures_verifiees');
            if (!File::exists($directory)) {
                File::makeDirectory($directory, 0755, true);
            }

            Log::info("Début de l'export des factures pour le dossier {$dossierId}");

            // Mise à jour de la progression
            $dossier->update([
                'verification_progress' => 92,
                'verification_step' => 'Préparation des données...'
            ]);

            // Récupérer toutes les structures avec formations sanitaires
            $structures = Structure::select(['id', 'nom_structure', 'distant_id'])
                ->where('niveau', 'Formation Sanitaire')
                ->get();

            if ($structures->isEmpty()) {
                Log::warning("Aucune structure de type 'Formation Sanitaire' trouvée");
            }

            $dataGlobal = [];
            $totalFactures = 0;
            $structuresTraitees = 0;

            foreach ($structures as $structure) {
                $factures = Facture::select([
                    'id',
                    'fis_id',
                    'statut_validation',
                    DB::raw("formation_sanitaire->>'id' as id_fs")
                ])
                    ->with('anomalies')
                    ->whereNotNull('statut_validation')
                    ->where('dossier_facture_id', $dossierId)
                    ->where(DB::raw("formation_sanitaire->>'id'"), $structure->distant_id)
                    ->get();

                $data = [];
                $data['id_structure'] = $structure->distant_id;
                $data['date_verification'] = Carbon::now()->format('Y-m-d H:i:s');
                $data['type_facture'] = 'standard';

                if ($factures->count() > 0) {
                    $totalFactures += $factures->count();
                    $structuresTraitees++;

                    foreach ($factures as $key => $facture) {
                        $observation = '';

                        // Construire les observations
                        if ($facture->anomalies && $facture->anomalies->count() > 0) {
                            $observation = $facture->anomalies->pluck('commentaire')->implode('; ');
                        } else {
                            $observation = 'RAS';
                        }

                        $data['factures'][$key] = [
                            'id_facture' => $facture->fis_id,
                            'status' => $facture->statut_validation,
                            'observations' => $observation,
                        ];
                    }
                } else {
                    $data['factures'] = [];
                }

                $dataGlobal[] = $data;
            }

            Log::info("Export: {$totalFactures} factures trouvées dans {$structuresTraitees} structures");

            // Mise à jour de la progression
            $dossier->update([
                'verification_progress' => 94,
                'verification_step' => 'Génération du fichier JSON...'
            ]);

            // Générer le JSON
            $facturesJson = json_encode($dataGlobal, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);

            if ($facturesJson == false) {
                throw new \Exception('Erreur lors de l\'encodage JSON : ' . json_last_error_msg());
            }

            $fileName = 'factures_standard_verifiees_' . $dossier->mois . '_' . $dossier->annee;
            $jsonFilePath = $directory . '/' . $fileName . '.json';

            // Écrire le fichier JSON
            if (file_put_contents($jsonFilePath, $facturesJson) == false) {
                throw new \Exception('Impossible d\'écrire le fichier JSON');
            }

            Log::info("Fichier JSON créé : {$jsonFilePath}");

            // Mise à jour de la progression
            $dossier->update([
                'verification_progress' => 96,
                'verification_step' => 'Compression du fichier...'
            ]);

            // Créer le ZIP
            $zipFilePath = $directory . '/' . $fileName . '.zip';

            try {
                Madzipper::make($zipFilePath)->add($jsonFilePath)->close();
                Log::info("Fichier ZIP créé : {$zipFilePath}");
            } catch (\Exception $e) {
                throw new \Exception('Erreur lors de la création du ZIP : ' . $e->getMessage());
            }

            // Supprimer le fichier JSON temporaire
            if (file_exists($jsonFilePath)) {
                unlink($jsonFilePath);
            }

            // Vérifier que le fichier ZIP existe et n'est pas vide
            if (!file_exists($zipFilePath) || filesize($zipFilePath) == 0) {
                throw new \Exception('Le fichier ZIP est invalide ou vide');
            }

            Log::info("Préparation de l'envoi vers FIS");

            // Mise à jour de la progression
            $dossier->update([
                'verification_progress' => 98,
                'verification_step' => 'Envoi vers FIS en cours...'
            ]);

            // Envoi vers FIS via ApiConnect
            $apiConnect = app(ApiConnect::class);

            $response = $apiConnect->callApi(
                'auth/controle/upload_vnf_results',
                'POST',
                $zipFilePath,
                [
                    'type' => 'standard',
                    'month' => (int) $dossier->mois,
                    'year' => (int) $dossier->annee,
                ]
            );

            // Nettoyer le fichier ZIP après envoi
            if (file_exists($zipFilePath)) {
                unlink($zipFilePath);
            }

            if ($response !== null && $response->getStatusCode() == 200) {
                $responseData = json_decode($response->getBody(), true);

                Log::info("Envoi vers FIS réussi", [
                    'dossier_id' => $dossierId,
                    'response' => $responseData
                ]);

                return [
                    'success' => true,
                    'message' => 'Envoi vers FIS réussi',
                    'data' => $responseData,
                    'total_factures' => $totalFactures,
                    'structures_traitees' => $structuresTraitees
                ];
            } else {
                $statusCode = $response ? $response->getStatusCode() : 'aucune réponse';
                $errorMessage = "Échec de l'envoi vers FIS (code: {$statusCode})";

                Log::error($errorMessage, [
                    'dossier_id' => $dossierId,
                    'status_code' => $statusCode
                ]);

                return [
                    'success' => false,
                    'message' => $errorMessage
                ];
            }

        } catch (\Exception $e) {
            Log::error("Erreur lors de l'export et envoi vers FIS", [
                'dossier_id' => $dossierId,
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString()
            ]);

            // Nettoyer les fichiers temporaires en cas d'erreur
            $fileName = 'factures_standard_verifiees_' . $dossier->mois . '_' . $dossier->annee;
            $jsonFilePath = storage_path('app/factures_verifiees/' . $fileName . '.json');
            $zipFilePath = storage_path('app/factures_verifiees/' . $fileName . '.zip');

            if (file_exists($jsonFilePath)) {
                @unlink($jsonFilePath);
            }
            if (file_exists($zipFilePath)) {
                @unlink($zipFilePath);
            }

            return [
                'success' => false,
                'message' => 'Erreur : ' . $e->getMessage()
            ];
        }
    }




    /**
     * @param Facture $facture
     * @param $type
     * @param $description
     * @param $decision
     * @param $commentaire
     * @return void
     */
    public static function creerAnomalie(Facture $facture, $type, $description, $decision, $commentaire)
    {
        //Pour prendre en compte la decision de STRFS de mettre toutes les factures rejetés a corriger, alors
        //si une facture est rejeté on enregistre son statut a corriger
        if($decision == "rejetee" && $facture->statut_validation == null){
            $decision = "a_corriger";
        }

        return $facture->anomalies()->create([
            'type_anomalie' => $type,
            'description_anomalie' => $description,
            'decision' => $decision,
            'commentaire' => $commentaire,
        ]);
    }

    /**
     * @param $nomFs
     * @return string
     *
     * Recuperation du type de formation sanitaire
     */
    private static function getTypeFs($nomFs){
        $typeFs = json_decode($nomFs);
        return $typeFs->type; //explode($typeFs->nom_fs, ' ')[0];
    }

    /**
     * Convertir l'âge du patient en années.
     *
     * @param string $agePatient
     * @return float
     */
    private static function convertirAgeEnAnnees($agePatient)
    {
        if (stripos($agePatient, 'jour') !== false) {
            $jours = (int) filter_var($agePatient, FILTER_SANITIZE_NUMBER_INT);
            return $jours / 365;
        } elseif (stripos($agePatient, 'mois') !== false) {
            $mois = (int) filter_var($agePatient, FILTER_SANITIZE_NUMBER_INT);
            return $mois / 12;
        } elseif (stripos($agePatient, 'an') !== false) {
            return (float) filter_var($agePatient, FILTER_SANITIZE_NUMBER_INT);
        }

        return 0;
    }

}
