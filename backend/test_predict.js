const axios = require('axios');
async function test() {
  try {
    let res = await axios.post("http://localhost:5001/api/predict", { symptoms: ["fever"] });
    console.log("fever:", res.data.disease);
    res = await axios.post("http://localhost:5001/api/predict", { symptoms: ["vomiting", "nausea"] });
    console.log("nausea:", res.data.disease);
  } catch (e) { console.error(e.response ? e.response.data : e.message); }
}
test();
