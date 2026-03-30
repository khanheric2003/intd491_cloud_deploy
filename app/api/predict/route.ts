import { NextRequest, NextResponse } from "next/server";

// Real logistic regression weights exported from trained sklearn model
const LR_COEFFICIENTS = [
  -0.6192135452982421,   // age
  0.13796377016946512,   // sex_male
  0.7883347675655511,    // priors_count
  0.03221666590915705,   // juv_fel_count
  -0.010377429937823184, // juv_misd_count
  0.08073173779230307,   // juv_other_count
  0.14324969223331926,   // charge_degree_felony
  -0.09109538971193913,  // age_cat_25-45
  0.11785333858268647,   // age_cat_Greater than 45
  -0.006213531674871821, // age_cat_Less than 25
];
const LR_INTERCEPT = 0.0002275770273049927;
const SCALER_MEAN = [34.44338667206806, 0.8091958679359935, 3.2616973870771724, 0.05651205185335224, 0.08932550131658902, 0.11079602997771926, 0.6481669029775167, 0.5815272432651408, 0.2039700222807373, 0.21450273445412193];
const SCALER_SCALE = [11.67534312741904, 0.39293500130594194, 4.7311749065891116, 0.39369991459648196, 0.4508689995484128, 0.4647585619260441, 0.4775422168374714, 0.4933085328732901, 0.40294695965044974, 0.4104769315879261];

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function predictRecidivism(
  age: number,
  sex: string,
  priorCrimes: number,
  juvenileFelonies: number,
  juvenileMisdemeanors: number,
  juvenileOther: number,
  chargeDegree: string,
): number {
  // Build feature vector matching prepare_features() output:
  // [age, sex_male, priors_count, juv_fel_count, juv_misd_count, juv_other_count,
  //  charge_degree_felony, age_cat_25-45, age_cat_Greater than 45, age_cat_Less than 25]
  const sexMale = sex === "Male" ? 1 : 0;
  const chargeFelony = chargeDegree === "Felony" ? 1 : 0;
  const ageCat2545 = (age >= 25 && age <= 45) ? 1 : 0;
  const ageCatGt45 = age > 45 ? 1 : 0;
  const ageCatLt25 = age < 25 ? 1 : 0;

  const features = [
    age,
    sexMale,
    priorCrimes,
    juvenileFelonies,
    juvenileMisdemeanors,
    juvenileOther,
    chargeFelony,
    ageCat2545,
    ageCatGt45,
    ageCatLt25,
  ];

  // StandardScaler transform: (x - mean) / scale
  const scaled = features.map((val, i) => (val - SCALER_MEAN[i]) / SCALER_SCALE[i]);

  // Dot product + intercept
  let logit = LR_INTERCEPT;
  for (let i = 0; i < scaled.length; i++) {
    logit += scaled[i] * LR_COEFFICIENTS[i];
  }

  return sigmoid(logit);
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const required = ["age", "sex", "prior_crimes", "juvenile_felonies", "juvenile_misdemeanors", "juvenile_other", "charge_degree"];
    for (const field of required) {
      if (!(field in body)) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        );
      }
    }

    if (body.age < 18 || body.age > 80) {
      return NextResponse.json(
        { error: "Age must be between 18 and 80" },
        { status: 400 }
      );
    }

    if (!["Male", "Female"].includes(body.sex)) {
      return NextResponse.json(
        { error: "Sex must be Male or Female" },
        { status: 400 }
      );
    }

    if (body.prior_crimes < 0 || body.prior_crimes > 40) {
      return NextResponse.json(
        { error: "Prior crimes must be between 0 and 40" },
        { status: 400 }
      );
    }

    if (body.juvenile_felonies < 0 || body.juvenile_felonies > 10) {
      return NextResponse.json(
        { error: "Juvenile felonies must be between 0 and 10" },
        { status: 400 }
      );
    }

    if (!["Felony", "Misdemeanor"].includes(body.charge_degree)) {
      return NextResponse.json(
        { error: "Charge degree must be Felony or Misdemeanor" },
        { status: 400 }
      );
    }

    const riskScore = predictRecidivism(
      body.age,
      body.sex,
      body.prior_crimes,
      body.juvenile_felonies,
      body.juvenile_misdemeanors,
      body.juvenile_other,
      body.charge_degree,
    );

    const riskLevel =
      riskScore > 0.6 ? "high" : riskScore > 0.3 ? "medium" : "low";

    return NextResponse.json({
      risk_score: riskScore,
      risk_level: riskLevel,
      model: "Logistic Regression",
      input: body,
    });
  } catch (error) {
    console.error("Prediction error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}
