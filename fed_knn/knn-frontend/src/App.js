import React, { useState } from 'react';
import axios from 'axios';

import './App.css';

// components folder
import SearchField from './components/SearchField';
import Node from './components/Node';
import Server from './components/Server';

// simple mui components
import ImportExportIcon from '@mui/icons-material/ImportExport';
import LoadingButton from '@mui/lab/LoadingButton';
import Waiting from './assets/turtle.png'
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';

function App() {
  const aggTechs = ['most_common', 'average_probabilities', 'bayesian'];
  const loadingIconsCount = 8;
  const loadingIcons = Array.from({ length: loadingIconsCount }, (_, index) => index + 1); // Create an array [1, 2, 3, 4, 5]

  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [aggLoading, setAggLoading] = useState(false);
  const [noInputAtAll, setNoInputAtAll] = useState(true);
  const [aggTech, setAggTech] = React.useState(aggTechs[0]);
  const [data, setData] = useState({
    'payload': {
        'server': {
            'class': 0,
            'spam': 50
        },
        'clients': [
            {'class': 0,'spam': 50, 'isPoisoned':false},
            {'class': 1,'spam': 50, 'isPoisoned':false},
            {'class': 0,'spam': 50, 'isPoisoned':false},
        ]
    },
    'status': 'success'
})

  
  const submitText=()=>{
    classifyInputRequest();
  }

  const classifyInputRequest = async () => {
    setLoading(true);
    setNoInputAtAll(false);
    try {
        const response = await axios.get(`http://127.0.0.1:5000/api/v1/classify?text=${inputText}`, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
        await new Promise(r => setTimeout(r, 1500));
        setData(response.data);
    } catch (error) {
        console.log(error);
    } finally {
        setLoading(false);
    }
};
  const setAggTechRequest = async (tech) => {
    setAggLoading(true);
    setAggTech(tech);
    try {
        const response = await axios.get(`http://127.0.0.1:5000/api/v1/setAggTech?aggTech=${tech}`, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
        setAggTech(response.data);
    } catch (error) {
        console.log(error);
    } finally {
      setAggLoading(false);
    }
  };
  return (
    <div className="App">
      <div className="content">
        <div className='content_input_container'>
            <div className="content_input">
              <SearchField submitText={submitText} aggLoading={aggLoading} inputText={inputText} setInputText={setInputText}/>
              <LoadingButton  loading={loading} variant="contained" sx={{ marginLeft:'10px', background:'var(--primary)', fontWeight:'bolder', color:'var(--primary-bg)', height:'50px' }} onClick={submitText}>{loading? 'Classify':'Classify'}</LoadingButton>
            </div>
            <div className='options_row'>
              <div style={{marginRight:'10px', display:'flex', justifyContent:'center', alignItems:'center'}}>
                <h4 style={{color:'var(--primary)'}}>Aggregation</h4>
              </div>
              <Stack direction="row" spacing={1}>
              {aggTechs.map((tech) => (
                  <Chip
                    key={tech}
                    sx={{ fontWeight: 'bold' }}
                    label={`${tech}`}
                    color="primary"
                    variant={aggTech === tech ? 'contained' : 'outlined'}
                    onClick={() => setAggTechRequest(tech)}
                  />
                ))}
              </Stack>
            </div>
        </div>
        {!noInputAtAll && <div className="content_visuals">
          <div className="server_content">
            {!loading && <Server isPoisoned={false} SpamPer={data['payload']['server']['spam']}/>}
          </div>
          <div className="loading_content">
          {loading && (
            <div className='import_icons_row'>
              {loadingIcons.map(index => (
              <div key={index} index={index} className='import_export_container'>
                <ImportExportIcon className='import_export'/>
              </div>
              ))}
            </div>
          )}
          </div>
          <div className='nodes_content'>
            {!loading && <div className="nodes_container">
              <Node isPoisoned={data['payload']['clients'][0]['isPoisoned']} SpamPer={data['payload']['clients'][0]['spam']}/>
              <Node isPoisoned={data['payload']['clients'][1]['isPoisoned']} SpamPer={data['payload']['clients'][1]['spam']}/>
              <Node isPoisoned={data['payload']['clients'][2]['isPoisoned']} SpamPer={data['payload']['clients'][2]['spam']}/>
            </div>}
          </div>
        </div>}
        {noInputAtAll && <div className='noInput'>
            <img src={Waiting} className="waiting_img" alt="poisoned" />
            <div className='noInput_message'>
              <h2>Let's start by writing a text to be <span className='primary-color'>classified</span>!</h2>
            </div>
          </div>}
      </div>
    </div>
  );
}

export default App;
