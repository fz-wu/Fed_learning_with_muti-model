import React from 'react';
import Poison from '../assets/poison.png'
import Valid from '../assets/valid.png'
import Spam from '../assets/spam_1.png'

export default function Node({ isPoisoned, SpamPer }) {
    return (
        <div className='node_container'>
            {isPoisoned && (
                <div className='poison_img_container'>
                    <img src={Poison} className="poison_img" alt="poisoned" />
                </div>
            )}
            <div className={`node ${SpamPer>=50 ? 'node_invalid' : 'node_valid'}`}>
                <div className='node_class node_ham_class' style={{height:`${100 - SpamPer}%`}}>
                    <div className='node_class_img'>
                        <img src={Valid} className="node_class_img" alt="Valid" /> 
                    </div>
                    <div className='node_class_per'>
                        {100 - SpamPer}%
                    </div>
                </div>
                <div className='node_class node_spam_class' style={{height:`${SpamPer}%`}}>
                    <div className='node_class_img'>
                        <img src={Spam} className="node_class_img" alt="poisoned" /> 
                    </div>
                    <div className='node_class_per'>
                        {SpamPer}%
                    </div>
                </div>
            </div>
            <div className='node_and_name'>
                <h4>Client</h4>
            </div>
        </div>
    );
}
