import React from 'react';

const Card = ({ children, className = '', hover = true, ...props }) => {
  const hoverClass = hover ? 'hover:shadow-md' : '';
  
  return (
    <div
      className={`bg-white rounded-lg border border-slate-200 shadow-sm transition duration-150 ease-out ${hoverClass} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

const CardHeader = ({ children, className = '' }) => (
  <div className={`px-6 py-4 border-b border-slate-200 ${className}`}>
    {children}
  </div>
);

const CardBody = ({ children, className = '' }) => (
  <div className={`px-6 py-4 ${className}`}>
    {children}
  </div>
);

const CardFooter = ({ children, className = '' }) => (
  <div className={`px-6 py-4 border-t border-slate-200 ${className}`}>
    {children}
  </div>
);

Card.Header = CardHeader;
Card.Body = CardBody;
Card.Footer = CardFooter;

export default Card;
