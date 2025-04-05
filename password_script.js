const express = require('express')
const mongoose = require('mongoose');
const path = require('path');
require('dotenv').config();

const app = express()
app.use(express.json());
const User=require('./password_modules/user');

mongoose.connect('mongodb://127.0.0.1:27017/myappdb', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  }).then(() => console.log('MongoDB connected'))
    .catch(err => console.error('MongoDB connection error:', err));

const port = 3000

app.use(express.static(path.join(__dirname, 'login_sign_page')));
app.use(express.static(path.join(__dirname, 'opening_page')));



app.post('/login',async(req,res)=>{
  const {email,password}=req.body;
  try{
    const user = await User.findOne({email});
    if (!user) return res.status(400).json({ message: 'User not found' });
    if (user.password !== password) {
      return res.status(400).json({ message: 'Incorrect password' });
    }
    res.status(200).json({ message: 'Login successful', user });  
  }catch (err) {
    res.status(500).json({ message: 'Server error' });
  } 
});

app.post('/users', async (req, res) => {
  try {
    const user = new User(req.body);
    const { name, email, password } = req.body;
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'Email is already in use' });
    }
    await user.save();
    res.status(201).json(user);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})