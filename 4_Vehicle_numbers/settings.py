"""Project settings. """

COMPARE_COEF = 1.95
MODELS_PATH = "./detectors/"
THRESHOLD = 0.5
LETTER_CLASSES = " 0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
VEHICLE_CLASSES = [3, 2, 4, 6, 8, 9]
CLASSES_NAMES = {
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
}
BRAND_CLASSES = [
    "AC-Cobra",
    "AEC",
    "AMC",
    "AMGeneral-HMMWV",
    "ARO",
    "Abarth",
    "Acura",
    "Acura-CL",
    "Acura-Integra",
    "Acura-RL",
    "Acura-RSX",
    "Acura-TL",
    "Acura-TLX",
    "Acura-TSX",
    "Acura-ZDX",
    "Adria",
    "Adria-Coral",
    "Adria-Matrix",
    "Aixam",
    "Ajokki",
    "AlexanderDennis",
    "AlfaBusz",
    "AlfaRomeo",
    "AlfaRomeo-145",
    "AlfaRomeo-146",
    "AlfaRomeo-155",
    "AlfaRomeo-166",
    "AlfaRomeo-2000",
    "AlfaRomeo-33",
    "AlfaRomeo-75",
    "AlfaRomeo-Brera",
    "AlfaRomeo-GT",
    "AlfaRomeo-GTV",
    "AlfaRomeo-Giulia",
    "AlfaRomeo-MiTo",
    "AlfaRomeo-Stelvio",
    "Alpine",
    "Alpine-A310",
    "Andecar-Viana",
    "Asia-Cosmos",
    "Asia-Rocsta",
    "Aston-Martin V8 Vantage",
    "AstonMartin",
    "AstonMartin-DB7",
    "AstonMartin-DB9",
    "AstonMartin-DBS",
    "AstonMartin-Rapide",
    "AstonMartin-V12 Vantage",
    "AstonMartin-Vanquish",
    "AstonMartin-Virage",
    "Audi",
    "Audi-200",
    "Audi-5000",
    "Audi-90",
    "Audi-Cabriolet",
    "Audi-Coupe",
    "Audi-Q2",
    "Audi-Q8",
    "Audi-Quattro",
    "Audi-RS Q3",
    "Audi-RS3",
    "Audi-RS5",
    "Audi-RS7",
    "Audi-S3",
    "Audi-S4",
    "Audi-S5 Sportback",
    "Audi-S7",
    "Audi-SQ7",
    "Audi-TT RS",
    "Audi-TTS",
    "Audi-V8",
    "Austin",
    "Austin-FX4",
    "Austin-Healey",
    "Austin-Mini",
    "Austin-Montego",
    "Austin-Seven",
    "Autobianchi",
    "Autobianchi-A112",
    "Autosan",
    "Avia",
    "Avia-A21",
    "Avia-A31",
    "Ayats",
    "Ayats-Atlas",
    "BAW",
    "BAW-Street",
    "BAZ",
    "BAZ-A081",
    "BAZBZKT",
    "BAZBZKT-5921",
    "BMC",
    "BMW",
    "BMW-1-Series M Coupe",
    "BMW-1600",
    "BMW-2-series",
    "BMW-2-series Active Tourer",
    "BMW-2002",
    "BMW-4-Series Gran Coupe",
    "BMW-6-Series Gran Turismo",
    "BMW-8-series",
    "BMW-E9",
    "BMW-Isetta",
    "BMW-M2",
    "BMW-M6 Gran Coupe",
    "BMW-X2",
    "BMW-X7",
    "BMW-Z4M",
    "BMW-Z8",
    "BYD",
    "Bedford",
    "BeifangBenchi",
    "BelAZ",
    "BelAZ-7555",
    "Bentley",
    "Bentley-Arnage",
    "Bentley-Brooklands",
    "Bentley-Continental Supersports",
    "Bentley-Flying Spur",
    "Bentley-Mulsanne",
    "Berkhof",
    "Berkhof-Excellence 2000",
    "Bertone",
    "Beulas",
    "Beulas-Stergo",
    "BlueBird",
    "Bogdan",
    "Bogdan-2111",
    "Bogdan-A093",
    "Bogdan-A144",
    "Bogdan-A201",
    "Bogdan-Lada 2310 Pickup",
    "Bogdan-Pадзимич A092",
    "Bollore-Bluecar",
    "Bova",
    "Brilliance",
    "Brilliance-M2",
    "Bronto-212180",
    "Bugatti",
    "Buick",
    "Buick-Century",
    "Buick-Eight",
    "Buick-Electra",
    "Buick-Encore",
    "Buick-LaCrosse",
    "Buick-LeSabre",
    "Buick-Park Avenue",
    "Buick-Regal",
    "Buick-Riviera",
    "Buick-Roadmaster",
    "Buick-Skylark",
    "Buick-Special",
    "Burstner",
    "CAMC",
    "CNHTC",
    "Cadillac",
    "Cadillac-ATS",
    "Cadillac-BLS",
    "Cadillac-CTS-V",
    "Cadillac-DTS",
    "Cadillac-Eldorado",
    "Cadillac-Fleetwood",
    "Cadillac-STS",
    "Cadillac-Series 62",
    "Cadillac-Seville",
    "Cadillac-XLR",
    "Cadillac-XT5",
    "Caetano",
    "Caetano-Optimo V",
    "Carado",
    "CaravansInternational",
    "Carrus",
    "Carrus-9700",
    "Carrus-Star",
    "Carthago",
    "Castrosúa",
    "Castrosúa-City Versus",
    "Caterham-Seven",
    "ChZSA",
    "Challenger",
    "ChanganChana",
    "ChanganChana-CS35",
    "ChanganChana-SC1030",
    "Chausson",
    "CherkassyBus",
    "CherkassyBus-Ataman A092H6",
    "CherkassyBus-Ataman A093",
    "Chery",
    "Chery-Beat",
    "Chery-Bonus",
    "Chery-Bonus 3",
    "Chery-CrossEastar",
    "Chery-E5",
    "Chery-Eastar",
    "Chery-Elara",
    "Chery-Jaggi",
    "Chery-M11",
    "Chery-QQ6",
    "Chery-Tiggo 2",
    "Chery-Tiggo 3",
    "Chery-Tiggo 5",
    "Chery-Very",
    "Chevrolet",
    "Chevrolet-3000-series",
    "Chevrolet-Astro",
    "Chevrolet-Avalanche",
    "Chevrolet-Bel Air",
    "Chevrolet-Cavalier",
    "Chevrolet-Chevelle",
    "Chevrolet-Chevy Van",
    "Chevrolet-Colorado",
    "Chevrolet-El Camino",
    "Chevrolet-Equinox",
    "Chevrolet-Evanda",
    "Chevrolet-HHR",
    "Chevrolet-Kalos",
    "Chevrolet-Lumina",
    "Chevrolet-Lumina APV",
    "Chevrolet-Master",
    "Chevrolet-Matiz",
    "Chevrolet-Monte Carlo",
    "Chevrolet-Nexia",
    "Chevrolet-Nova",
    "Chevrolet-Nubira",
    "Chevrolet-S-10",
    "Chevrolet-Tacuma",
    "Chevrolet-Tracker",
    "Chevrolet-Trans Sport",
    "Chevrolet-Traverse",
    "Chevrolet-Trax",
    "Chevrolet-Viva",
    "Chevrolet-Volt",
    "Chrysler",
    "Chrysler-200",
    "Chrysler-Cirrus",
    "Chrysler-Intrepid",
    "Chrysler-LHS",
    "Chrysler-LeBaron",
    "Chrysler-Neon",
    "Chrysler-New Yorker",
    "Chrysler-Newport",
    "Chrysler-Stratus",
    "Citroen",
    "Citroen-Acadiane",
    "Citroen-Ami",
    "Citroen-C15",
    "Citroen-C25",
    "Citroen-C3 Aircross",
    "Citroen-C3 Pluriel",
    "Citroen-C4 AirCross",
    "Citroen-C6",
    "Citroen-C8",
    "Citroen-CX",
    "Citroen-DS5",
    "Citroen-Dyane",
    "Citroen-Evasion",
    "Citroen-GS",
    "Citroen-HY",
    "Citroen-Mehari",
    "Citroen-Nemo",
    "Citroen-Traction Avant",
    "Citroen-Visa",
    "Citroen-XM",
    "Citroen-ZX",
    "Citroеn",
    "Citroеn-Acadiane",
    "Citroеn-C3 Pluriel",
    "Citroеn-C6",
    "Citroеn-C8",
    "Citroеn-Nemo",
    "Citroеn-Visa",
    "DAF",
    "DAF-45",
    "DAF-LF",
    "DAF-Leyland 400",
    "DS-3",
    "DS-4",
    "DS-5",
    "DS-7 Crossback",
    "DWHower",
    "Dacia",
    "Dacia-1300",
    "Dacia-1310",
    "Dacia-Dokker",
    "Dacia-Lodgy",
    "Dacia-Sandero Stepway",
    "Dacia-Solenza",
    "Dadi",
    "Daewoo",
    "Daewoo-BH115",
    "Daewoo-BH117H",
    "Daewoo-BH120F",
    "Daewoo-BM090",
    "Daewoo-Damas",
    "Daewoo-Kalos",
    "Daewoo-Lacetti",
    "Daewoo-Lestar",
    "Daewoo-Lublin",
    "Daewoo-Magnus",
    "Daewoo-Novus",
    "Daewoo-Prima",
    "Daewoo-Tacuma",
    "Daewoo-Ultra",
    "Daewoo-Winstorm",
    "Daihatsu",
    "Daihatsu-Applause",
    "Daihatsu-Atrai",
    "Daihatsu-Be-Go",
    "Daihatsu-Charade",
    "Daihatsu-Copen",
    "Daihatsu-Cuore",
    "Daihatsu-Gran Move",
    "Daihatsu-Hijet",
    "Daihatsu-Materia",
    "Daihatsu-Mira",
    "Daihatsu-Move",
    "Daihatsu-Pyzar",
    "Daihatsu-Sirion",
    "Daihatsu-Terios Kid",
    "Daihatsu-YRV",
    "Daimler",
    "Daimler-DS420",
    "Daimler-Double Six",
    "Daimler-Super V8",
    "Daimler-V8",
    "Datsun",
    "Datsun-240Z",
    "Datsun-Laurel",
    "DeLoreanMotorCompany",
    "DeTomaso",
    "Delta",
    "Derways",
    "Derways-Aurora",
    "Derways-Shuttle",
    "Dethleffs",
    "Dethleffs-Globetrotter",
    "Dodge",
    "Dodge-Avenger",
    "Dodge-Coronet",
    "Dodge-Dakota",
    "Dodge-Dart",
    "Dodge-Dynasty",
    "Dodge-Journey",
    "Dodge-Magnum",
    "Dodge-Mini Ram Van",
    "Dodge-Ram Van",
    "Dodge-Sprinter",
    "Dodge-Stealth",
    "Dodge-Viper",
    "Dodge-WC-series",
    "DongFeng",
    "DongFeng-H30 Cross",
    "DongFeng-KC",
    "DongFeng-S30",
    "Doninvest",
    "Drögmöller",
    "Drögmöller-E330",
    "Drögmöller-EuroComet",
    "EOS",
    "EOS-200",
    "EOS-90",
    "ERF",
    "Eagle",
    "Eagle-Vision",
    "Elnagh",
    "Emgrand-EC7-RV",
    "ErnstAuwärter",
    "ErnstAuwärter-Teamstar",
    "EuraMobil",
    "Excalibur-Phaeton",
    "Excalibur-Phantom",
    "FAW",
    "FAW-Besturn B50",
    "FAW-J5",
    "FAW-J6",
    "FAW-Jiabao",
    "FAW-Vita",
    "FAW-Xiao Jiefang",
    "FIAT",
    "FIAT-1100",
    "FIAT-124",
    "FIAT-126",
    "FIAT-128",
    "FIAT-131",
    "FIAT-500L",
    "FIAT-Aegea",
    "FIAT-Barchetta",
    "FIAT-Cinquecento",
    "FIAT-Coupe",
    "FIAT-Croma",
    "FIAT-Freemont",
    "FIAT-Fullback",
    "FIAT-Idea",
    "FIAT-Marea",
    "FIAT-Multipla",
    "FIAT-Palio",
    "FIAT-Punto Evo",
    "FIAT-Qubo",
    "FIAT-Regata",
    "FIAT-Ritmo",
    "FIAT-Sedici",
    "FIAT-Siena",
    "FIAT-Talento",
    "FIAT-Tempra",
    "FIAT-Ulysse",
    "FSM",
    "FSO-125",
    "FSO-Warszawa",
    "Ferqui",
    "Ferrari",
    "Ferrari-308",
    "Ferrari-328",
    "Ferrari-348",
    "Ferrari-360",
    "Ferrari-365",
    "Ferrari-488",
    "Ferrari-512",
    "Ferrari-550",
    "Ferrari-575",
    "Ferrari-599",
    "Ferrari-612",
    "Ferrari-812",
    "Ferrari-California",
    "Ferrari-Dino",
    "Ferrari-F12",
    "Ferrari-F12tdf",
    "Ferrari-F355",
    "Ferrari-FF",
    "Ferrari-GTC4",
    "Ferrari-LaFerrari",
    "Fisker",
    "Ford",
    "Ford-Aerostar",
    "Ford-Anglia",
    "Ford-B-Max",
    "Ford-Bronco",
    "Ford-Capri",
    "Ford-Contour",
    "Ford-Crown",
    "Ford-Deluxe",
    "Ford-E-series",
    "Ford-Econoline",
    "Ford-Econovan",
    "Ford-Escort",
    "Ford-Excursion",
    "Ford-Explorer",
    "Ford-F-100",
    "Ford-F-250",
    "Ford-F-350",
    "Ford-F-450",
    "Ford-F-650",
    "Ford-Fairlane",
    "Ford-Fiesta",
    "Ford-Flex",
    "Ford-Freda",
    "Ford-Freestyle",
    "Ford-Galaxie",
    "Ford-Grand",
    "C-Max",
    "Ford-Model A",
    "Ford-Model T",
    "Ford-Orion",
    "Ford-Puma",
    "Ford-Spectron",
    "Ford-StreetKa",
    "Ford-Tempo",
    "Ford-Thunderbird",
    "Ford-Tourneo",
    "Ford-Tourneo Courier",
    "Ford-Tourneo Custom",
    "Ford-Transit Custom",
    "Ford-V8",
    "Ford-Windstar",
    "Foton",
    "Foton-Aumark",
    "Foton-Forland",
    "FoxBus-22501",
    "Freightliner",
    "Freightliner-Argosy",
    "Freightliner-FL-series",
    "Freightliner-FLB",
    "Freightliner-FLD",
    "Freightliner-Sprinter",
    "GAZ",
    "GAZ-12",
    "GAZ-13",
    "GAZ-14",
    "GAZ-22",
    "GAZ-22177",
    "GAZ-22177",
    "GAZ-2310",
    "GAZ-2331",
    "GAZ-24-12",
    "GAZ-24-13",
    "GAZ-31022",
    "GAZ-3111",
    "GAZ-3274",
    "GAZ-33086",
    "GAZ-3325",
    "GAZ-4301",
    "GAZ-51",
    "GAZ-5903",
    "GAZ-5903",
    "GAZ-63",
    "GAZ-67",
    "GAZ-704",
    "GAZ-93",
    "GAZ-CA3-3503",
    "GAZ-CA3-35071",
    "GAZ-CA3-3511",
    "GAZ-CA3-4509",
    "GAZ-M1",
    "GMC",
    "GMC-Acadia",
    "GMC-Envoy",
    "GMC-Jimmy",
    "GMC-Safari",
    "GMC-Savana",
    "GMC-Savana Van 2012",
    "GMC-Sierra",
    "GMC-Terrain",
    "GMC-Vandura",
    "GMC-Yukon",
    "GMC-Yukon XL",
    "Geely",
    "Geely-Atlas",
    "Geely-Emgrand 7",
    "Geely-GC6",
    "Geely-LC Cross",
    "Geely-Otaka",
    "Geely-SC7",
    "Geely-SL",
    "Genesis-G70",
    "Genesis-G80",
    "Genesis-G90",
    "Geo",
    "Geo-Metro",
    "GiottiLine",
    "Globecar",
    "GolAZ",
    "GolAZ-5251",
    "GolAZ-5291",
    "GolAZ-6228",
    "GolAZ-AKA-5225",
    "GolAZ-ЛиA3-5256",
    "GoldenDragon",
    "GoldenDragon-XML6112",
    "GoldenDragon-XML6126",
    "GreatWall",
    "GreatWall-Deer",
    "GreatWall-Haval H3",
    "GreatWall-Hover M2",
    "GreatWall-Hover M4",
    "GreatWall-Sailor",
    "GreatWall-Voleex C30",
    "GreatWall-Wingle",
    "Gräf&Stift",
    "HUMMER",
    "HUMMER-H1",
    "HUMMER-H2",
    "Haargaz",
    "Hafei",
    "Hafei-Brio",
    "Hafei-Lobo",
    "Haima",
    "Haima-3",
    "HalAZ",
    "Haval-H2",
    "Haval-H9",
    "HawtaiHuatai",
    "Hess",
    "Higer",
    "Higer-A80",
    "Higer-KLQ6109",
    "Higer-KLQ6119TQ",
    "Higer-KLQ6129",
    "Higer-KLQ6728",
    "Higer-KLQ6840",
    "Higer-KLQ6891",
    "Hino",
    "Hino-700",
    "Hino-Dutro",
    "Hino-FN",
    "Hino-FS",
    "Hino-FW",
    "Hispano",
    "Honda",
    "Honda-Accord",
    "Honda-Acty",
    "Honda-CR-Z",
    "Honda-CRX",
    "Honda-Capa",
    "Honda-City",
    "Honda-Civic",
    "Honda-Concerto",
    "Honda-Crossroad",
    "Honda-Domani",
    "Honda-Elysion",
    "Honda-FR-V",
    "Honda-Fit",
    "Honda-Fit Shuttle",
    "Honda-Freed",
    "Honda-Freed Spike",
    "Honda-Inspire",
    "Honda-Life",
    "Honda-Logo",
    "Honda-Mobilio",
    "Honda-Mobilio Spike",
    "Honda-Orthia",
    "Honda-Partner",
    "Honda-Rafaga",
    "Honda-Ridgeline",
    "Honda-S-MX",
    "Honda-S2000",
    "Honda-Saber",
    "Honda-Shuttle",
    "Honda-Torneo",
    "Honda-Valkyrie",
    "Honda-Vezel",
    "Huanghai",
    "Hymer-Camp",
    "Hymer-Mobil",
    "Hyundai",
    "Hyundai-Aero Express",
    "Hyundai-Aero Queen",
    "Hyundai-Aero Space",
    "Hyundai-Atos",
    "Hyundai-Atos Prime",
    "Hyundai-Azera",
    "Hyundai-Genesis Coupe",
    "Hyundai-Gold",
    "Hyundai-Grace",
    "Hyundai-Grand i10",
    "Hyundai-H100",
    "Hyundai-H200",
    "Hyundai-HD1000",
    "Hyundai-HD170",
    "Hyundai-HD270",
    "Hyundai-HD65",
    "Hyundai-Ioniq",
    "Hyundai-Kona",
    "Hyundai-Lavita",
    "Hyundai-Mighty",
    "Hyundai-Pony",
    "Hyundai-Santamo",
    "Hyundai-Super Aero City",
    "Hyundai-Tiburon Turbulence",
    "Hyundai-Universe",
    "Hyundai-Universe Xpress",
    "Hyundai-Veracruz",
    "Hyundai-Verna",
    "Hyundai-XG",
    "IFA-L60",
    "IFA-W50",
    "IFA-W50L",
    "IFA-W50LA",
    "Ikarus",
    "Ikarus-396",
    "Ikarus-415",
    "Ikarus-435",
    "Ikarus-55",
    "Ikarus-C56",
    "Ikarus-E-series",
    "Indcar",
    "Infiniti",
    "Infiniti-I-series",
    "Infiniti-Q30",
    "Infiniti-QX4",
    "Innocenti-Mini",
    "International",
    "International-DuraStar",
    "Irisbus",
    "Irisbus-Crossway",
    "Irisbus-Magelys",
    "Irizar",
    "Irizar-Century",
    "Irizar-Century III",
    "Irizar-PB",
    "Irizar-i6",
    "Isuzu",
    "Isuzu-CYZ",
    "Isuzu-D-Max",
    "Isuzu-F-series",
    "Isuzu-Fargo",
    "Isuzu-Gemini",
    "Isuzu-Giga",
    "Isuzu-Gigamax",
    "Isuzu-MU Wizard",
    "Isuzu-Novo",
    "Isuzu-Rodeo",
    "Isuzu-Turkuaz",
    "Isuzu-V-series",
    "Isuzu-VehiCross",
    "Iveco",
    "Iveco-EuroTrakker",
    "Iveco-LMV",
    "Iveco-PowerDaily",
    "Iveco-TurboStar",
    "Iveco-TurboZeta",
    "Iveco-VM 90",
    "Iveco-Zeta",
    "Izh",
    "Izh-27151",
    "Izh-27156",
    "Izh-49",
    "JAC",
    "JAC-HFC1045",
    "JAC-HK6120 KY",
    "JAC-Refine",
    "JAC-Rein",
    "JAC-S5",
    "JMC",
    "Jaguar",
    "Jaguar-E-Type",
    "Jaguar-I-Pace",
    "Jaguar-Mark-series",
    "Jaguar-SS",
    "Jaguar-Sovereign",
    "Jaguar-XE",
    "Jaguar-XJR",
    "Jaguar-XJS",
    "Jaguar-XK120",
    "Jaguar-XK140",
    "Jaguar-XK150",
    "Jawa",
    "Jeep",
    "Jeep-CJ-series",
    "Jeep-Commander",
    "Jeep-Liberty",
    "Jeep-Patriot",
    "Jeep-Renegade",
    "Jelcz",
    "Jensen",
    "JinBei-Haise",
    "Jonckheere",
    "Jonckheere-Mistral",
    "KATO-NK-series",
    "KAZ-608",
    "KAvZ",
    "KAvZ-39765",
    "KAvZ-651",
    "KAvZ-685",
    "KIA",
    "KIA-AM818 Cosmos",
    "KIA-Avella Delta",
    "KIA-Besta",
    "KIA-Clarus",
    "KIA-Combi",
    "KIA-K5",
    "KIA-Morning",
    "KIA-New Cosmos",
    "KIA-Niro",
    "KIA-Pregio",
    "KIA-Pride",
    "KIA-Retona",
    "KIA-Sedona",
    "KIA-Sephia",
    "KIA-Shuma",
    "KIA-Shuma II",
    "KIA-Sportage Grand",
    "KIA-Stinger",
    "KIA-Stonic",
    "KamAZ",
    "KamAZ-4325",
    "KamAZ-43255",
    "KamAZ-4326",
    "KamAZ-4350 ",
    "KamAZ-43501",
    "KamAZ-43502",
    "KamAZ-4510",
    "KamAZ-45141",
    "KamAZ-45142",
    "KamAZ-45144",
    "KamAZ-4528",
    "KamAZ-53228",
    "KamAZ-53504",
    "KamAZ-5387",
    "KamAZ-5460",
    "KamAZ-6350",
    "KamAZ-65111",
    "KamAZ-65201",
    "KamAZ-65206",
    "KamAZ-6522",
    "KamAZ-65221",
    "KamAZ-65222",
    "KamAZ-65225",
    "KamAZ-6540",
    "KamAZ-42111",
    "KamAZ-Mагнолия",
    "Karosa",
    "Karosa-C734",
    "Karosa-C954E",
    "Karosa-LC735",
    "Karsan",
    "Kenworth",
    "Kenworth-T800",
    "KhAZ",
    "KingLong",
    "KingLong-XMQ6127",
    "KingLong-XMQ6129",
    "Knaus",
    "KrAZ",
    "KrAZ-255",
    "KrAZ-256",
    "KrAZ-257",
    "KrAZ-260",
    "KrAZ-5233",
    "KrAZ-6322",
    "KrAZ-6443",
    "KrAZ-6505",
    "Kravtex",
    "Kravtex-Credo EC12",
    "Kravtex-Credo Econell",
    "Kuban",
    "Kuban-Г1",
    "Kutter",
    "LAZ",
    "LAZ-4207",
    "LAZ-5252",
    "LAZ-697",
    "LAZ-A141",
    "LDV-Convoy",
    "LIAZ",
    "LMC",
    "LTI",
    "LTI-TX4",
    "Lahti-Eagle",
    "Laika",
    "Lamborghini",
    "Lamborghini-Murcielago",
    "Lamborghini-Urus",
    "Lancia",
    "Lancia-Beta",
    "Lancia-Dedra",
    "Lancia-Fulvia",
    "Lancia-Kappa",
    "Lancia-Lybra",
    "Lancia-Musa",
    "Lancia-Phedra",
    "Lancia-Thema",
    "Lancia-Thesis",
    "Lancia-Voyager",
    "Lancia-Y10",
    "Land",
    "LandRover",
    "LandRover-Series III",
    "Lexus-HS",
    "Lexus-IS-F",
    "Lexus-RC",
    "Lexus-RC-F",
    "LiAZ-4292",
    "LiAZ-5250",
    "Liebherr-LTM",
    "Lifan",
    "Lifan-Cebrium",
    "Lincoln",
    "Lincoln-Aviator",
    "Lincoln-Continental",
    "Lincoln-MKT",
    "Lincoln-MKX",
    "Lincoln-MKZ",
    "LiveZone",
    "Lotus",
    "Lotus-Elise",
    "Lotus-Esprit",
    "Lotus-Europa",
    "Lotus-Evora",
    "Lotus-Exige",
    "LuAZ-967",
    "LuAZ-969",
    "Luxgen",
    "MAN",
    "MAN-F8",
    "MAN-Lion's City",
    "MAN-Lion's Classic",
    "MAN-Lion's Regio",
    "MAN-Lion's Star",
    "MAN-Lion's Top Coach",
    "MAN-M2000",
    "MAN-M90",
    "MAN-ME2000",
    "MAN-NG",
    "MAN-S2000",
    "MAN-SL",
    "MAN-SU",
    "MAN-Volkswagen G90",
    "MARZ-42191",
    "MARZ-5277",
    "MAZ",
    "MAZ-105",
    "MAZ-152",
    "MAZ-215",
    "MAZ-231",
    "MAZ-251",
    "MAZ-256",
    "MAZ-4570",
    "MAZ-500",
    "MAZ-504",
    "MAZ-5309",
    "MAZ-5335",
    "MAZ-5340",
    "MAZ-5429",
    "MAZ-5433",
    "MAZ-5549",
    "MAZ-6317",
    "MAZ-6425",
    "MAZ-6501",
    "MAZ-6516",
    "MAZ-6517",
    "MAZ-8378",
    "MAZ-9380",
    "MAZ-9386",
    "MAZ-9758",
    "MG",
    "MG-A",
    "MG-Midget",
    "MG-TC",
    "MG-ZR",
    "MG-ZT",
    "MINI",
    "MINI-Convertible",
    "MINI-Coupe",
    "MINI-Moke",
    "MINI-Paceman",
    "MZKT",
    "Mack",
    "Mack-Pinnacle",
    "MagirusMagirusDeutz",
    "MagirusMagirusDeutz-D-series",
    "MagirusMagirusDeutz-Eckhauber",
    "MagirusMagirusDeutz-MK-series",
    "Mahindra",
    "Marcopolo-Andare",
    "Marcopolo-Bravis",
    "Marcopolo-Real",
    "Maruti-800",
    "Maserati",
    "Maserati-GranCabrio",
    "Maserati-Spyder",
    "Matra",
    "Maxus",
    "Maybach",
    "Maybach-57",
    "Maybach-57S",
    "Maybach-62S",
    "Mazda",
    "Mazda-121",
    "Mazda-929",
    "Mazda-Atenza",
    "Mazda-Axela",
    "Mazda-B-series",
    "Mazda-Biante",
    "Mazda-Bongo Brawny",
    "Mazda-CX-3",
    "Mazda-E-series",
    "Mazda-Eunos Roadster",
    "Mazda-Luce",
    "Mazda-MX-3",
    "Mazda-MX-6",
    "Mazda-Millenia",
    "Mazda-Proceed",
    "Mazda-Proceed Marvie",
    "Mazda-Protege",
    "Mazda-RX-7",
    "Mazda-Roadster",
    "Mazda-Verisa",
    "Mazda-Xedos 6",
    "Mazda-Xedos 9",
    "Mazda-Ẽfini MPV",
    "McLaren",
    "McLaren-570",
    "McLaren-650S",
    "McLaren-675LT",
    "McLaren-720S",
    "McLaren-MP4-12C",
    "McLouis",
    "Mercedes",
    "MercedesBenz",
    "MercedesBenz-170",
    "MercedesBenz-Arocs",
    "MercedesBenz-CLC-Klasse",
    "MercedesBenz-Citan",
    "MercedesBenz-Conecto",
    "MercedesBenz-Conecto G",
    "MercedesBenz-E-Klasse",
    "All-Terrain",
    "MercedesBenz-O305",
    "MercedesBenz-O307",
    "MercedesBenz-O309D",
    "MercedesBenz-O325",
    "MercedesBenz-O345",
    "MercedesBenz-O403",
    "MercedesBenz-O404",
    "MercedesBenz-O405G",
    "MercedesBenz-O407",
    "MercedesBenz-O560 Intouro",
    "MercedesBenz-SLC-Klasse",
    "MercedesBenz-SLR McLaren",
    "MercedesBenz-Type 180",
    "MercedesBenz-Unimog",
    "MercedesBenz-Vaneo",
    "MercedesBenz-X-Klasse",
    "Mercury",
    "Mercury-Cougar",
    "Mercury-Grand Marquis",
    "Mercury-Mariner",
    "Mercury-Sable",
    "Mercury-Topaz",
    "Mercury-Villager",
    "Merkavim-Mars",
    "Mitsubishi",
    "Mitsubishi-3000GT",
    "Mitsubishi-Chariot Grandis",
    "Mitsubishi-Delica D:5",
    "Mitsubishi-Diamante",
    "Mitsubishi-Dion",
    "Mitsubishi-Eclipse Cross",
    "Mitsubishi-Fuso Rosa",
    "Mitsubishi-Fuso The Great",
    "Mitsubishi-L400",
    "Mitsubishi-Lancer Cargo",
    "Mitsubishi-Legnum",
    "Mitsubishi-Libero",
    "Mitsubishi-Minica",
    "Mitsubishi-Minicab",
    "Mitsubishi-Mirage Dingo",
    "Mitsubishi-Pajero Junior",
    "Mitsubishi-Pajero Mini",
    "Mitsubishi-Shogun",
    "Mitsubishi-Sigma",
    "Mitsubishi-Space Runner",
    "Mitsubishi-Toppo",
    "Mitsubishi-eK",
    "Mitsubishi-i",
    "Mitsuoka",
    "Mitsuoka-Galue",
    "Morgan",
    "Morgan-Plus 4",
    "Morris",
    "Morris-Minor",
    "MoskvitchAZLK",
    "MoskvitchAZLK-2142",
    "MoskvitchAZLK-2335",
    "MoskvitchAZLK-2901",
    "MoskvitchAZLK-400",
    "MoskvitchAZLK-402",
    "MoskvitchAZLK-410",
    "MoskvitchAZLK-423",
    "MoskvitchAZLK-426",
    "MoskvitchAZLK-427",
    "Mudan",
    "Multicar-M25",
    "Multicar-M26",
    "NSU",
    "Nash",
    "NefAZ",
    "NefAZ-8122",
    "NefAZ-8332",
    "NefAZ-8560",
    "NefAZ-9334",
    "NefAZ-VDL",
    "Neman",
    "Neman-4202",
    "Neoplan",
    "Neoplan-Euroliner",
    "Neoplan-Jetliner",
    "Neoplan-Spaceliner",
    "Neoplan-Starliner",
    "Neoplan-Tourliner",
    "Neoplan-Transliner",
    "Niesmann+Bischoff",
    "Nissan",
    "Nissan-100NX",
    "Nissan-200SX",
    "Nissan-240SX",
    "Nissan-300ZX",
    "Nissan-370Z",
    "Nissan-Almera Tino",
    "Nissan-Armada",
    "Nissan-Bassara",
    "Nissan-Caravan",
    "Nissan-Caravan Elgrand",
    "Nissan-Cima",
    "Nissan-Diesel Big Thumb",
    "Nissan-Dualis",
    "Nissan-Eco",
    "Nissan-Expert",
    "Nissan-Fairlady",
    "Nissan-Figaro",
    "Nissan-Frontier",
    "Nissan-Fuga",
    "Nissan-Gloria",
    "Nissan-Interstar",
    "Nissan-Lafesta",
    "Nissan-Largo",
    "Nissan-NV200",
    "Nissan-NV200 Vanette",
    "Nissan-Pick Up",
    "Nissan-Pixo",
    "Nissan-Prairie",
    "Nissan-Prairie Joy",
    "Nissan-Presea",
    "Nissan-Primastar",
    "Nissan-Primera Camino",
    "Nissan-Quest",
    "Nissan-R",
    "Nissan-Rasheen",
    "Nissan-Rogue",
    "Nissan-Sunny AD",
    "Nissan-Tino",
    "Nissan-Titan",
    "Nissan-Trade",
    "Nissan-Urvan",
    "Nissan-Vanette Largo",
    "Nissan-Versa",
    "Nissan-Xterra",
    "Nissan-e-NV200",
    "Noge",
    "Noge-Touring Star",
    "Nysa-522",
    "Oldsmobile",
    "Oldsmobile-98",
    "Oldsmobile-Aurora",
    "Oldsmobile-Cutlass",
    "Oltcit",
    "Opel",
    "Opel-Adam",
    "Opel-Ampera",
    "Opel-Astra TwinTop",
    "Opel-Commodore",
    "Opel-Crossland X",
    "Opel-Frontera Sport",
    "Opel-GT",
    "Opel-Grandland X",
    "Opel-Insignia Country Tourer",
    "Opel-Manta",
    "Opel-Mokka X",
    "Opel-Monterey",
    "Opel-Monza",
    "Opel-Senator",
    "Opel-Signum",
    "Opel-Sintra",
    "Opel-Speedster",
    "Opel-Super 6",
    "Orlandi",
    "Otokar",
    "Otokar-Sultan",
    "Otoyol",
    "Otoyol-M29 City",
    "PAZ",
    "PAZ-3204",
    "PAZ-32051",
    "PAZ-3206",
    "PAZ-3237",
    "PAZ-3742",
    "PAZ-Vector Next",
    "Packard",
    "Panhard",
    "Panhard-PL17",
    "Pegaso",
    "Peterbilt",
    "Peterbilt-387",
    "Peugeot",
    "Peugeot-1007",
    "Peugeot-104",
    "Peugeot-108",
    "Peugeot-203",
    "Peugeot-204",
    "Peugeot-206",
    "Peugeot-207 CC",
    "Peugeot-304",
    "Peugeot-305",
    "Peugeot-307 CC",
    "Peugeot-308 CC",
    "Peugeot-309",
    "Peugeot-4008",
    "Peugeot-403",
    "Peugeot-404",
    "Peugeot-504",
    "Peugeot-505",
    "Peugeot-605",
    "Peugeot-806",
    "Peugeot-Bipper",
    "Peugeot-Bipper Tepee",
    "Peugeot-Expert Tepee",
    "Peugeot-J5",
    "Peugeot-J7",
    "Peugeot-J9",
    "Peugeot-Traveller",
    "Piaggio-Ape",
    "Piaggio-Porter",
    "Pilote",
    "Plaxton",
    "Plymouth",
    "Plymouth-Barracuda",
    "Plymouth-Road Runner",
    "Plymouth-Voyager",
    "PolskiFiat-125p",
    "Pontiac",
    "Pontiac-Bonneville",
    "Pontiac-Fiero",
    "Pontiac-GTO",
    "Pontiac-Grand Am",
    "Pontiac-Grand Prix",
    "Pontiac-Solstice",
    "Pontiac-Sunfire",
    "Pontiac-Trans Sport",
    "Porsche",
    "Porsche-356",
    "Porsche-718 Boxster",
    "Porsche-718 Cayman",
    "Porsche-914",
    "Porsche-924",
    "Porsche-928",
    "Porsche-944",
    "Porsche-968",
    "Porsche-Carrera GT",
    "Praga",
    "Proton",
    "Proton-400-series",
    "PskovAvto",
    "PskovAvto-AПB-У-01",
    "PskovAvto-AПB-У-03",
    "PskovAvto-AПB-У-05",
    "Pössl",
    "RAF",
    "RAF-22031",
    "RAF-2915",
    "Ram",
    "Ravon-Gentra",
    "Ravon-Nexia R3",
    "Ravon-R2",
    "Ravon-R4",
    "Renault",
    "Renault-11",
    "Renault-12",
    "Renault-16",
    "Renault-18",
    "Renault-25",
    "Renault-4CV",
    "Renault-6",
    "Renault-8",
    "Renault-9",
    "Renault-Clio Symbol",
    "Renault-D-series",
    "Renault-Dauphine",
    "Renault-Dokker",
    "Renault-Estafette",
    "Renault-FR1",
    "Renault-Fuego",
    "Renault-Grand Kangoo",
    "Renault-Grand Modus",
    "Renault-Kerax",
    "Renault-Lodgy",
    "Renault-Logan Van",
    "Renault-Magnum Mack 430",
    "Renault-Major",
    "Renault-Manager",
    "Renault-Mascott",
    "Renault-Midliner",
    "Renault-Midlum",
    "Renault-Modus",
    "Renault-Rodeo",
    "Renault-Safrane",
    "Renault-Scenic RX4",
    "Renault-Talisman",
    "Renault-Twizy Z",
    "Renault-Vel Satis",
    "Renault-Zoe",
    "Riley",
    "Rimor",
    "Roewe",
    "RollerTeam",
    "Rolls",
    "RollsRoyce",
    "RollsRoyce-Corniche",
    "RollsRoyce-Dawn",
    "RollsRoyce-Phantom Coupe",
    "RollsRoyce-Phantom Drophead Coupe",
    "RollsRoyce-Silver Cloud",
    "RollsRoyce-Silver Seraph",
    "RollsRoyce-Silver Shadow",
    "RollsRoyce-Silver Spirit",
    "RollsRoyce-Silver Wraith",
    "Rover",
    "Rover-200-series",
    "Rover-25",
    "Rover-400-series",
    "Rover-45",
    "Rover-600-series",
    "Rover-800-series",
    "Rover-Mini",
    "Rover-P4",
    "Rover-Streetwise",
    "Ruta",
    "Ruta-25",
    "Ruta-A048",
    "Ruta-CПB-15",
    "Ruta-CПB-17",
    "Rába",
    "SAAB",
    "SAAB-96",
    "SAAB-99",
    "SAMCO",
    "SAZ",
    "SAZ-NP37",
    "SEAT",
    "SEAT-124",
    "SEAT-127",
    "SEAT-1430",
    "SEAT-600",
    "SEAT-850",
    "SEAT-Altea",
    "SEAT-Altea Freetrack",
    "SEAT-Altea XL",
    "SEAT-Arona",
    "SEAT-Arosa",
    "SEAT-Exeo",
    "SEAT-Inca",
    "SEAT-Marbella",
    "SEAT-Panda",
    "SEAT-Terra",
    "SMA",
    "SOR",
    "Samsung",
    "Saturn",
    "Saturn-S-series",
    "Saturn-VUE",
    "Scania",
    "Scania-112",
    "Scania-113",
    "Scania-II-series",
    "Scania-OmniExpress",
    "Scania-S-series",
    "Scania-T-series",
    "Scania-Touring HD (Higer A80T)",
    "Scion",
    "Scion-tC",
    "Scion-xA",
    "Scion-xB",
    "SeAZ-C-3",
    "SemAR-3234",
    "SemAR-3280",
    "Setra",
    "Setra-500-series",
    "Shaanxi",
    "Shaanxi-SX3310-series",
    "Shaolin",
    "ShuangHuan-Sceo",
    "SimAZ",
    "Simca",
    "Simca-1000",
    "Simca-Aronde",
    "Sisu",
    "Skoda",
    "Skoda-100",
    "Skoda-1000",
    "Skoda-105",
    "Skoda-110",
    "Skoda-1202",
    "Skoda-14",
    "Skoda-706",
    "Skoda-Citigo",
    "Skoda-Forman",
    "Skoda-Karoq",
    "Skoda-Pickup",
    "Skoda-Praktik",
    "Skoda-Roomster Scout",
    "Smart-Roadster Coupe",
    "Smit-Orion",
    "Solaris",
    "Solaris-Urbino 12",
    "Spyker",
    "SsangYong-Korando Family",
    "SsangYong-Korando Sports",
    "SsangYong-Musso",
    "SsangYong-Musso Sports",
    "SsangYong-Rodius",
    "SsangYong-Stavic",
    "SsangYong-Tivoli",
    "SsangYong-TransStar",
    "Star",
    "Sterling",
    "Steyr",
    "Steyr-92",
    "Subaru",
    "Subaru-BRZ",
    "Subaru-Domingo",
    "Subaru-Impreza XV",
    "Subaru-Justy",
    "Subaru-Legacy Lancaster",
    "Subaru-Legacy Outback",
    "Subaru-Leone",
    "Subaru-Libero",
    "Subaru-R2",
    "Subaru-SVX",
    "Subaru-Sambar",
    "Subaru-WRX",
    "Sunbeam",
    "SunlongShenLong",
    "Sunsundegui",
    "Sunsundegui-SC7",
    "Sunsundegui-Sideral",
    "Sunsundegui-Sideral 2000",
    "Suzuki",
    "Suzuki-Aerio",
    "Suzuki-Baleno",
    "Suzuki-Celerio",
    "Suzuki-Cultus",
    "Suzuki-Cultus Crescent",
    "Suzuki-Every",
    "Suzuki-Grand Escudo",
    "Suzuki-Jimny Sierra",
    "Suzuki-Jimny Wide",
    "Suzuki-Kei",
    "Suzuki-Kizashi",
    "Suzuki-SJ-series",
    "Suzuki-Samurai",
    "Suzuki-Sidekick",
    "Suzuki-Swift J",
    "Säffle",
    "TAM",
    "TAZTrnava",
    "TREKOL-39294",
    "TVR",
    "TagAZ",
    "TagAZ-Aquila",
    "TagAZ-C190",
    "TagAZ-Road Partner",
    "TagAZ-Vega",
    "TajikistanChAZ",
    "TajikistanChAZ-3205",
    "Talbot",
    "Talbot-Horizon",
    "Tata",
    "Tatra",
    "Tatra-148",
    "Tatra-Jamal",
    "TemSA",
    "TemSA-MD9",
    "TemSA-Opalin",
    "TemSA-Prestij",
    "TemSA-Safari",
    "TemSA-Safir",
    "Tesla",
    "Tesla-Model 3",
    "Tesla-Model X",
    "Thaco",
    "Tofas-Dogan",
    "Tofas-Kartal",
    "Tofas-Murat 131",
    "Tonar-9523",
    "Toyota",
    "Toyota-Aqua",
    "Toyota-Aristo",
    "Toyota-Avensis Verso",
    "Toyota-Belta",
    "Toyota-Brevis",
    "Toyota-Cami",
    "Toyota-Cavalier",
    "Toyota-Celica XX",
    "Toyota-Celsior",
    "Toyota-Coaster",
    "Toyota-Commuter",
    "Toyota-Corolla Altis",
    "Toyota-Corolla Ceres",
    "Toyota-Corolla FX",
    "Toyota-Corolla II",
    "Toyota-Corolla Rumion",
    "Toyota-Corona Exiv",
    "Toyota-Corona Mark II",
    "Toyota-Crown Athlete",
    "Toyota-Crown Majesta",
    "Toyota-Curren",
    "Toyota-Cynos",
    "Toyota-Duet",
    "Toyota-Echo",
    "Toyota-Estima Emina",
    "Toyota-Estima Lucida",
    "Toyota-Grand Hiace",
    "Toyota-Granvia",
    "Toyota-Hiace Regius",
    "Toyota-Innova",
    "Toyota-Isis",
    "Toyota-Land Cruiser Cygnus",
    "Toyota-Land Cruiser II",
    "Toyota-MR-2",
    "Toyota-MR-S",
    "Toyota-Mark II Blit",
    "Toyota-Mark II Qualis",
    "Toyota-Master Ace Surf",
    "Toyota-Paseo",
    "Toyota-Picnic",
    "Toyota-Porte",
    "Toyota-Prius PLUS",
    "Toyota-Prius α",
    "Toyota-ProAce",
    "Toyota-Progres",
    "Toyota-Quick Delivery",
    "Toyota-Regius",
    "Toyota-Rush",
    "Toyota-Sai",
    "Toyota-Scepter",
    "Toyota-Sera",
    "Toyota-Sienta",
    "Toyota-Soarer",
    "Toyota-Sparky",
    "Toyota-Sprinter Cielo",
    "Toyota-Sprinter Marino",
    "Toyota-Sprinter Truen",
    "Toyota-Tacoma",
    "Toyota-Tercel",
    "Toyota-Touring Hiace",
    "Toyota-Urban Cruiser",
    "Toyota-Vanguard",
    "Toyota-Vellfire",
    "Toyota-Verossa",
    "Toyota-Verso-S",
    "Toyota-Vios",
    "Toyota-Will Cypha",
    "Toyota-Will VS",
    "Toyota-Will Vi",
    "Toyota-Yaris Verso",
    "Toyota-bB Open Deck",
    "Toyota-iQ",
    "Toyota-xA",
    "Trabant",
    "Triumph",
    "Triumph-Herald",
    "Triumph-Spitfire",
    "Triumph-Stag",
    "Triumph-TR3",
    "Triumph-TR4",
    "Triumph-TR6",
    "UAZ",
    "UAZ-23602 Cargo",
    "UAZ-3153",
    "UAZ-3160",
    "UAZ-3162",
    "UAZ-3164 Patriot Sport",
    "UAZ-AC-01",
    "UAZ-AC-B1",
    "UAZ-Profi",
    "UAZ-ПДП",
    "UNVI",
    "Ugarte",
    "UralAZ",
    "UralAZ-3255",
    "UralAZ-375",
    "UralAZ-4320 Next",
    "UralAZ-43204",
    "UralAZ-44202",
    "UralAZ-5323",
    "UralAZ-63685",
    "UralAZ-H3AC-4951",
    "UralAZ-HефA3-4211",
    "VAZLada",
    "VAZLada-0-1932-01",
    "VAZLada-0-19321",
    "VAZLada-210934",
    "VAZLada-21108 Premier",
    "VAZLada-2123",
    "VAZLada-21708 Premier",
    "VAZLada-2172 Priora Coupe",
    "VAZLada-2329",
    "VAZLada-Vesta Cross",
    "VAZLada-XRAY Cross",
    "VDL-Citea",
    "VDL-Futura",
    "VIS-2346",
    "VIS-23461",
    "VIS-2349",
    "VanHool",
    "VanHool-AG300",
    "VanHool-T8",
    "VanHool-T815",
    "VanHool-T9",
    "VanHool-T915",
    "VanHool-T916",
    "VanHool-T917",
    "VanHool-TX-series",
    "Vauxhall",
    "Vauxhall-Astra",
    "Vauxhall-Corsa",
    "Vauxhall-Insignia",
    "Vauxhall-Vectra",
    "Vauxhall-Zafira",
    "Vest",
    "Volgabus",
    "Volgabus-52701",
    "Volgabus-52702",
    "Volgabus-5285",
    "Volgabus-6270",
    "Volgabus-6271",
    "Volkswagen",
    "Volkswagen-Arteon",
    "Volkswagen-Buggy",
    "Volkswagen-Bus",
    "Volkswagen-California",
    "Volkswagen-Corrado",
    "Volkswagen-CrossGolf",
    "Volkswagen-CrossPolo",
    "Volkswagen-Eos",
    "Volkswagen-EuroVan",
    "Volkswagen-Fox",
    "Volkswagen-Golf Country",
    "Volkswagen-Golf Sportsvan",
    "Volkswagen-Karmann-Ghia",
    "Volkswagen-Lupo",
    "Volkswagen-Passat Alltrack",
    "Volkswagen-Pointer",
    "Volkswagen-Santana",
    "Volkswagen-T-Roc",
    "Volkswagen-Teramont",
    "Volkswagen-Typ 181",
    "Volkswagen-Typ 3",
    "Volvo",
    "Volvo-140-series",
    "Volvo-340-series",
    "Volvo-360",
    "Volvo-440",
    "Volvo-460",
    "Volvo-480",
    "Volvo-760",
    "Volvo-960",
    "Volvo-9700",
    "Volvo-9900",
    "Volvo-C70",
    "Volvo-F10",
    "Volvo-FE",
    "Volvo-P1800",
    "Volvo-PV-series",
    "Volvo-S60 Cross Country",
    "Volvo-S70",
    "Volvo-S90",
    "Volvo-V60",
    "Volvo-V60 Cross Country",
    "Volvo-V90",
    "Volvo-V90 Cross Country",
    "Volvo-XC40",
    "Wartburg-311",
    "Wiesmann",
    "Wiima-K202",
    "Willys",
    "Willys-MB",
    "Wrightbus",
    "Wuling-Hongguang",
    "Wuling-Sunshine",
    "XCMG",
    "XCMG-QY25",
    "Yaxing",
    "YouYi",
    "YouYi-ZGT6710",
    "YouYi-ZGT6718",
    "Yuejin-NJ1041",
    "Yuejin-NJ1080",
    "Yutong",
    "Yutong-ZK6118HGA",
    "Yutong-ZK6119HA",
    "Yutong-ZK6122",
    "Yutong-ZK6737D",
    "Yutong-ZK6899H",
    "ZAZ",
    "ZAZ-11024",
    "ZAZ-966",
    "ZAZ-Forza",
    "ZAZ-Lanos-Фургон",
    "ZAZ-Vida",
    "ZIL",
    "ZIL-3250",
    "ZIL-41047",
    "ZIL-4329",
    "ZIL-4334",
    "ZIL-4415",
    "ZIL-4421",
    "ZIL-AMУP-5313",
    "ZIL-CAA3-4545",
    "ZIL-MM3-4506",
    "ZIL-MM3-4952",
    "ZIL-MM3-555",
    "ZIS",
    "ZX-Landmark",
    "Zastava",
    "Zastava-1100",
    "Zastava-750",
    "Zastava-Koral",
    "Zastava-Skala",
    "Zastava-Yugo 45A",
    "ZhongTong",
    "ZhongTong-LCK6127",
    "ZhongTong-LCK6605",
    "Zonda",
    "Zoomlion-QY-series",
    "Zotye-T600",
]
