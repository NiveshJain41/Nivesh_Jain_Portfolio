import React, { useState } from 'react';
import emailjs from 'emailjs-com';
import { useToast } from '@/hooks/use-toast';

const SERVICE_ID = 'service_vknuc34';
const TEMPLATE_ID = 'template_dr0ysz6';
const PUBLIC_KEY = '_yK48S1FmZF0kikJo';

const ContactForm = () => {
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    const now = new Date().toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      dateStyle: 'medium',
      timeStyle: 'short',
    });

    const templateParams = {
      from_name: formData.name,                 // used for reply name
      from_email: formData.email,              // used for reply email
      message: formData.message,               // message content
      time: now,                                // timestamp
      title: "Portfolio Contact Request",       // optional title field
      name: formData.name,                      // name shown in template
      email: formData.email                     // email shown in template
    };

    try {
      await emailjs.send(SERVICE_ID, TEMPLATE_ID, templateParams, PUBLIC_KEY);

      toast({
        title: 'Message sent successfully!',
        description: "Thanks for reaching out. I'll get back to you soon.",
      });

      setFormData({ name: '', email: '', message: '' });
    } catch (error) {
      toast({
        title: 'Failed to send message.',
        description: 'Something went wrong. Please try again.',
      });
      console.error('EmailJS Error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="bg-navy-800/50 p-6 rounded-xl backdrop-blur-sm">
      <div className="mb-6">
        <label htmlFor="name" className="block mb-2 font-medium">Name</label>
        <input
          type="text"
          id="name"
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
          className="w-full px-4 py-3 bg-navy-700/50 border border-navy-600 rounded-md 
                     focus:outline-none focus:ring-2 focus:ring-purple/50 text-black placeholder-gray-600"
          placeholder="Your name"
        />
      </div>

      <div className="mb-6">
        <label htmlFor="email" className="block mb-2 font-medium">Email</label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          required
          className="w-full px-4 py-3 bg-navy-700/50 border border-navy-600 rounded-md 
                     focus:outline-none focus:ring-2 focus:ring-purple/50 text-black placeholder-gray-600"
          placeholder="your.email@example.com"
        />
      </div>

      <div className="mb-6">
        <label htmlFor="message" className="block mb-2 font-medium">Message</label>
        <textarea
          id="message"
          name="message"
          value={formData.message}
          onChange={handleChange}
          required
          rows={5}
          className="w-full px-4 py-3 bg-navy-700/50 border border-navy-600 rounded-md 
                     focus:outline-none focus:ring-2 focus:ring-purple/50 text-black placeholder-gray-600 resize-none"
          placeholder="Your message here..."
        ></textarea>
      </div>

      <button
        type="submit"
        disabled={isSubmitting}
        className={`btn-primary w-full flex items-center justify-center ${
          isSubmitting ? 'opacity-70 cursor-not-allowed' : ''
        }`}
      >
        {isSubmitting ? 'Sending...' : 'Send Message'}
      </button>
    </form>
  );
};

export default ContactForm;
