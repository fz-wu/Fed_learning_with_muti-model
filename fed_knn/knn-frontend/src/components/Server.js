import React from 'react';
import Poison from '../assets/poison.png'
import Valid from '../assets/valid.png'
import Spam from '../assets/spam_1.png'

export default function Server({ isPoisoned, SpamPer }) {
  return (
    <div className='node_and_name'>
        <div className='server_container'>
            {isPoisoned && (
                <div className='poison_img_container'>
                    <img src={Poison} className="poison_img" alt="poisoned" />
                </div>
            )}
            <div className={`server ${SpamPer>=50 ? 'server_invalid' : 'server_valid'}`}>
                <div className='server_class server_ham_class' style={{width:`${100 - SpamPer}%`, flexDirection:'column'}}>
                    <img src={Valid} alt="Valid" />  {100 - SpamPer}%
                </div>
                <div className='server_class server_spam_class' style={{width:`${SpamPer}%`, flexDirection:'column'}}>
                    <img src={Spam} alt="Valid" />  {SpamPer}%
                </div>
            </div>
            
        </div>
        <h3>Server</h3>
    </div>
  );
}
