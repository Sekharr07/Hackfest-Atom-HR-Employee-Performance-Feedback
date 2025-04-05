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

const user_1={
  body:{
    name:"Rio",
    email:"rajarsee5@gmail.com",
    password:"randrio@2005"
  }
}

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'login_sign_page/index.html'));
})

app.post('/testserver',async(req,res)=>{
  const user_1={
    body:{
      name:"Rio",
      email:"rajarsee5@gmail.com",
      password:"randrio@2005"
    }
  }
  try{
    const user = new User(user_1.body);
    await user.save();
    res.status(201).json(user);
  }
  catch (err) {
    res.status(400).json({ error: err.message });
  }
})



app.post('/users', async (req, res) => {
  try {
    const user = new User(req.body);
    await user.save();
    res.status(201).json(user);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})