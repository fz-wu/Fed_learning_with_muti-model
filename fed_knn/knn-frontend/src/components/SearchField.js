import axios from 'axios';
import { motion } from 'framer-motion';
import Paper from '@mui/material/Paper';
import InputBase from '@mui/material/InputBase';
import IconButton from '@mui/material/IconButton';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ShuffleIcon from '@mui/icons-material/Shuffle';
import Tooltip from '@mui/material/Tooltip';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';

export default function SearchField({submitText, inputText, aggLoading, setInputText}) {

  const preventDefault = (e)=>{
    if (e.key === 'Enter') {
      e.preventDefault();  // Prevents the form from being submitted
      submitText();
    }
  }

  const handleChange = (e)=>{
    setInputText(e.target.value);
  }

  const clearInput = ()=>{
    setInputText("");
  }

  const getRandomInput = async () => {

    try {
        const response = await axios.get(`http://127.0.0.1:5000/api/v1/randomInput`);
        console.log(response.data);
        setInputText(response.data);
    } catch (error) {
        console.log(error);
    } finally {

    }
  };

  return (
    <Paper
      component="form"
      sx={{ p: '2px 4px', display: 'flex', alignItems: 'center', width: '100%', backgroundColor: 'var(--secondary-bg)', border: '1px solid var(--primary)' }}
    >
      <IconButton
        className="search_input_icon"
        sx={{
          opacity: inputText.length > 0 ? 0 : 1,
          p: '10px',
          color: 'var(--primary)',  // Ensure this CSS variable is defined in your styles
          transition: 'opacity 300ms ease-in-out'
        }}
        aria-label="menu"
      >
        <ChevronRightIcon />
      </IconButton>
      <IconButton
        className="search_input_icon"
        sx={{ position: 'absolute', p: '10px', color: 'var(--gray)' }}
        aria-label="menu"
        onClick={clearInput}
      >
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: inputText.length > 0 ? 1 : 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
        >
          <DeleteForeverIcon />
        </motion.div>
      </IconButton>
      <InputBase
        sx={{ ml: 1, flex: 1 , color:'white'}}
        placeholder="Enter a Text to Classify"
        value={inputText}
        inputProps={{ 'aria-label': 'search google maps' }}
        onKeyDownCapture={preventDefault}
        onChange={handleChange}
        disabled={aggLoading}
      />
      <Tooltip title="Generate Random Input">
        <IconButton disabled={aggLoading} type="button" sx={{ p: '10px', color:'var(--primary)'}} aria-label="search" onClick={getRandomInput}>
          <ShuffleIcon />
        </IconButton>
      </Tooltip>
    </Paper>
    
  );
}
